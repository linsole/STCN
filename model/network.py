"""
network.py - The core of the neural network
Defines the structure and memory operations
Modifed from STM: https://github.com/seoungwugoh/STM

The trailing number of a variable usually denote the stride
e.g. f16 -> encoded features with stride 16
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from model.modules import *


class Decoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.compress = ResBlock(1024, 512)
        self.up_16_8 = UpsampleBlock(512, 512, 256) # 1/16 -> 1/8
        self.up_8_4 = UpsampleBlock(256, 256, 256) # 1/8 -> 1/4

        self.pred = nn.Conv2d(256, 1, kernel_size=(3,3), padding=(1,1), stride=1)

    def forward(self, f16, f8, f4):
        #: f16:4,1024,24,24    f8: 4,512,48,48    f4: 4,256,96,96
        #: notice the use of f8 and f4 at decoding stages, this is the 'skip connection' 
        #: between encoded feature map and the output of the previous stage
        x = self.compress(f16) #: 4,512,24,24
        x = self.up_16_8(f8, x) #: 4,256,48,48
        x = self.up_8_4(f4, x) #: 4,256,96,96

        x = self.pred(F.relu(x)) #: 4,1,96,96
        
        x = F.interpolate(x, scale_factor=4, mode='bilinear', align_corners=False) #: 4,1,384,384
        return x


class MemoryReader(nn.Module):
    def __init__(self):
        super().__init__()
 
    def get_affinity(self, mk, qk):
        B, CK, T, H, W = mk.shape
        mk = mk.flatten(start_dim=2)
        qk = qk.flatten(start_dim=2)

        # See supplementary material
        a_sq = mk.pow(2).sum(1).unsqueeze(2)
        ab = mk.transpose(1, 2) @ qk

        affinity = (2*ab-a_sq) / math.sqrt(CK)   # B, THW, HW
        
        # softmax operation; aligned the evaluation style
        maxes = torch.max(affinity, dim=1, keepdim=True)[0]
        x_exp = torch.exp(affinity - maxes)
        x_exp_sum = torch.sum(x_exp, dim=1, keepdim=True)
        affinity = x_exp / x_exp_sum 

        return affinity

    def readout(self, affinity, mv, qv):
        B, CV, T, H, W = mv.shape

        mo = mv.view(B, CV, T*H*W) 
        mem = torch.bmm(mo, affinity) # Weighted-sum B, CV, HW
        mem = mem.view(B, CV, H, W)

        mem_out = torch.cat([mem, qv], dim=1)

        return mem_out


class STCN(nn.Module):
    def __init__(self, single_object):
        super().__init__()
        self.single_object = single_object

        self.key_encoder = KeyEncoder()
        if single_object:
            self.value_encoder = ValueEncoderSO() 
        else:
            self.value_encoder = ValueEncoder() 

        # Projection from f16 feature space to key space
        self.key_proj = KeyProjection(1024, keydim=64)

        # Compress f16 a bit to use in decoding later on
        self.key_comp = nn.Conv2d(1024, 512, kernel_size=3, padding=1)

        self.memory = MemoryReader()
        self.aspp = ASPP(1024, 1024)
        self.refine = RefinementModule()
        self.scm = SCM()
        self.decoder = Decoder()

    def aggregate(self, prob):
        new_prob = torch.cat([
            torch.prod(1-prob, dim=1, keepdim=True),
            prob
        ], 1).clamp(1e-7, 1-1e-7)
        logits = torch.log((new_prob /(1-new_prob)))
        return logits

    def encode_key(self, frame): 
        # input: b*t*c*h*w    frame: 4,3,3,384,384
        b, t = frame.shape[:2] #: b:4, t:3

        #: flatten the data to 2-dimension and encode the key
        f16, f8, f4 = self.key_encoder(frame.flatten(start_dim=0, end_dim=1))
        #: f16: 12,1024,24,24    f8: 12,512,48,48    f4: 12,256,96,96
        k16 = self.key_proj(f16) #: k16: 12,64,24,24
        f16_thin = self.key_comp(f16) #: f16_thin: 12,512,24,24

        # B*C*T*H*W  
        #: after flatten, reconstruct the tensor, 'contiguous' makes sure 
        #: that semantic and physical memory are consistent
        k16 = k16.view(b, t, *k16.shape[-3:]).transpose(1, 2).contiguous() #: k16: 4,64,3,24,24

        # B*T*C*H*W
        f16_thin = f16_thin.view(b, t, *f16_thin.shape[-3:]) #: f16_thin: 4,3,512,24,24
        f16 = f16.view(b, t, *f16.shape[-3:]) #: 4,3,1024,24,24
        f8 = f8.view(b, t, *f8.shape[-3:]) #: 4,3,512,48,48
        f4 = f4.view(b, t, *f4.shape[-3:]) #: 4,3,256,96,96

        return k16, f16_thin, f16, f8, f4

    def encode_value(self, frame, kf16, mask, other_mask=None): 
        # Extract memory key/value for a frame
        #: frame: 4,3,384,384    kf16: 4,1024,24,24    mask: 4,1,384,384(other_mask)
        if self.single_object:
            f16 = self.value_encoder(frame, kf16, mask)
        else:
            f16 = self.value_encoder(frame, kf16, mask, other_mask) #: 4,512,24,24
        return f16.unsqueeze(2) # B*512*T*H*W #: 4,512,1,24,24

    def segment(self, qk16, qv16, qf8, qf4, mk16, mv16, mask, other_mask=None, selector=None): 
        # q - query, m - memory
        #: qk16: 4,64,24,24    qv16: 4,512,24,24    qf8: 4,512,48,48    qf4: 4,256,96,96
        #: mk16: 4,64,1(2),24,24    mv16: 4,2,512,1(2),24,24
        #: similarity between mk16 and qk16 implies reusing of encoded query key as memory key
        # qv16 is f16_thin above
        affinity = self.memory.get_affinity(mk16, qk16) #: 4,576,576
        
        if self.single_object:
            readout = self.aspp(self.memory.readout(affinity, mv16, qv16))
            readout = self.scm(readout, mask)
            logits = self.decoder(readout, qf8, qf4) #: 8,1,384,384
            prob = torch.sigmoid(logits) #: 8,1,384,384
            prob = self.refine(prob)
        else:
            readout1 = self.aspp(self.memory.readout(affinity, mv16[:,0], qv16)) #: 4,1024,24,24
            readout2 = self.aspp(self.memory.readout(affinity, mv16[:,1], qv16)) #: 4,1024,24,24
            readout1 = self.scm(readout1, mask)
            readout2 = self.scm(readout2, other_mask)
            logits = torch.cat([
                self.decoder(readout1, qf8, qf4),
                self.decoder(readout2, qf8, qf4),
            ], 1) #: 4,2,384,384

            prob = torch.sigmoid(logits) #: 4,2,384,384
            
            #: add refinement module before soft aggregation
            prob = torch.cat([self.refine(prob[:,0].unsqueeze(1)), self.refine(prob[:,1].unsqueeze(1))], 1)

            #: Attention! It's important that this procedure is placed at last, i.e *after refinement module*, 
            #: otherwise we may incorrectly produce a *second mask* when dealing with single object segmentation, 
            #: which is possible during main training even this is under the *not self.single_object* condition
            prob = prob * selector.unsqueeze(2).unsqueeze(2) #: selector.unsqueeze(2).unsqueeze(2): 4,2,1,1

        logits = self.aggregate(prob) #: single: 8,2,384,384    multiple: 4,3,384,384
        prob = F.softmax(logits, dim=1)[:, 1:] #: single: 8,1,384,384    multiple: 4,2,384,384

        return logits, prob

    def forward(self, mode, *args, **kwargs):
        #: go to different functions according to the passed string argument
        if mode == 'encode_key':
            return self.encode_key(*args, **kwargs)
        elif mode == 'encode_value':
            return self.encode_value(*args, **kwargs)
        elif mode == 'segment':
            return self.segment(*args, **kwargs)
        else:
            raise NotImplementedError


