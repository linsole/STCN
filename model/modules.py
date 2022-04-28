"""
modules.py - This file stores the rathering boring network blocks.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

from model import mod_resnet
from model import cbam


class ResBlock(nn.Module):
    def __init__(self, indim, outdim=None):
        super(ResBlock, self).__init__()
        if outdim == None:
            outdim = indim
        if indim == outdim:
            self.downsample = None
        else:
            self.downsample = nn.Conv2d(indim, outdim, kernel_size=3, padding=1)
 
        self.conv1 = nn.Conv2d(indim, outdim, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(outdim, outdim, kernel_size=3, padding=1)
 
    def forward(self, x):
        r = self.conv1(F.relu(x))
        r = self.conv2(F.relu(r))
        
        if self.downsample is not None:
            x = self.downsample(x)

        return x + r


class FeatureFusionBlock(nn.Module):
    def __init__(self, indim, outdim):
        super().__init__()

        self.block1 = ResBlock(indim, outdim)
        self.attention = cbam.CBAM(outdim)
        self.block2 = ResBlock(outdim, outdim)

    def forward(self, x, f16):
        x = torch.cat([x, f16], 1)
        x = self.block1(x)
        r = self.attention(x)
        x = self.block2(x + r)

        return x


# Single object version, used only in static image pretraining
# This will be loaded and modified into the multiple objects version later (in stage 1/2/3)
# See model.py (load_network) for the modification procedure
class ValueEncoderSO(nn.Module):
    def __init__(self):
        super().__init__()

        resnet = mod_resnet.resnet18(pretrained=True, extra_chan=1)
        self.conv1 = resnet.conv1
        self.bn1 = resnet.bn1
        self.relu = resnet.relu  # 1/2, 64
        self.maxpool = resnet.maxpool

        self.layer1 = resnet.layer1 # 1/4, 64
        self.layer2 = resnet.layer2 # 1/8, 128
        self.layer3 = resnet.layer3 # 1/16, 256

        self.fuser = FeatureFusionBlock(1024 + 256, 512)

    def forward(self, image, key_f16, mask):
        # key_f16 is the feature from the key encoder
        #: during pre-trainging, image: 8*3*384*384, mask: 8*1*384*384,
        #: so cat([image, mask], 1) leads to 8*4*384*384
        f = torch.cat([image, mask], 1)

        x = self.conv1(f)
        x = self.bn1(x)
        x = self.relu(x)   # 1/2, 64
        x = self.maxpool(x)  # 1/4, 64
        x = self.layer1(x)   # 1/4, 64
        x = self.layer2(x) # 1/8, 128
        x = self.layer3(x) # 1/16, 256

        x = self.fuser(x, key_f16)

        return x


# Multiple objects version, used in other times
class ValueEncoder(nn.Module):
    def __init__(self):
        super().__init__()

        resnet = mod_resnet.resnet18(pretrained=True, extra_chan=2)
        self.conv1 = resnet.conv1
        self.bn1 = resnet.bn1
        self.relu = resnet.relu  # 1/2, 64
        self.maxpool = resnet.maxpool

        self.layer1 = resnet.layer1 # 1/4, 64
        self.layer2 = resnet.layer2 # 1/8, 128
        self.layer3 = resnet.layer3 # 1/16, 256

        self.fuser = FeatureFusionBlock(1024 + 256, 512)

    def forward(self, image, key_f16, mask, other_masks):
        # key_f16 is the feature from the key encoder

        f = torch.cat([image, mask, other_masks], 1)

        x = self.conv1(f)
        x = self.bn1(x)
        x = self.relu(x)   # 1/2, 64
        x = self.maxpool(x)  # 1/4, 64
        x = self.layer1(x)   # 1/4, 64
        x = self.layer2(x) # 1/8, 128
        x = self.layer3(x) # 1/16, 256

        x = self.fuser(x, key_f16)

        return x
 

#: the structure of value and key encoder are quite similar to each other
class KeyEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        resnet = models.resnet50(pretrained=True)
        self.conv1 = resnet.conv1
        self.bn1 = resnet.bn1
        self.relu = resnet.relu  # 1/2, 64
        self.maxpool = resnet.maxpool

        self.res2 = resnet.layer1 # 1/4, 256
        self.layer2 = resnet.layer2 # 1/8, 512
        self.layer3 = resnet.layer3 # 1/16, 1024

    def forward(self, f):
        x = self.conv1(f) 
        x = self.bn1(x)
        x = self.relu(x)   # 1/2, 64
        x = self.maxpool(x)  # 1/4, 64
        f4 = self.res2(x)   # 1/4, 256
        f8 = self.layer2(f4) # 1/8, 512
        f16 = self.layer3(f8) # 1/16, 1024

        return f16, f8, f4


#: 'Upsample' means it scales *up* the spatial dimention
class UpsampleBlock(nn.Module):
    def __init__(self, skip_c, up_c, out_c, scale_factor=2):
        super().__init__()
        self.skip_conv = nn.Conv2d(skip_c, up_c, kernel_size=3, padding=1)
        self.out_conv = ResBlock(up_c, out_c)
        self.scale_factor = scale_factor

    def forward(self, skip_f, up_f):
        x = self.skip_conv(skip_f)
        x = x + F.interpolate(up_f, scale_factor=self.scale_factor, mode='bilinear', align_corners=False)
        x = self.out_conv(x)
        return x


class KeyProjection(nn.Module):
    def __init__(self, indim, keydim):
        super().__init__()
        self.key_proj = nn.Conv2d(indim, keydim, kernel_size=3, padding=1)

        nn.init.orthogonal_(self.key_proj.weight.data)
        nn.init.zeros_(self.key_proj.bias.data)
    
    def forward(self, x):
        return self.key_proj(x)


#: add an ASPP module and apply it to the readout value of memory reader
class ASPP(nn.Module):
    def __init__(self, in_channel=1024, depth=1024):
        super(ASPP, self).__init__()
        self.atrous_block2 = nn.Conv2d(in_channel, depth, 3, 1, padding=2, dilation=2)
        self.atrous_block4 = nn.Conv2d(in_channel, depth, 3, 1, padding=4, dilation=4)
        self.atrous_block8 = nn.Conv2d(in_channel, depth, 3, 1, padding=8, dilation=8)
        self.conv_1x1_output = nn.Conv2d(depth * 3, depth, 1, 1)

    def forward(self, x):
        atrous_block2 = self.atrous_block2(x)
        atrous_block4 = self.atrous_block4(x)
        atrous_block8 = self.atrous_block8(x)
        net = self.conv_1x1_output(torch.cat([atrous_block2, atrous_block4, atrous_block8], dim=1))
        return net


#: add refinement module, take feature map before soft aggrefation as input
#: we only deal with multiple objects for now
class RefinementModule(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 64, kernel_size=3, padding=1, stride=2)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1, stride=2)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, padding=1, stride=2)
        self.compress = ResBlock(256, 128)
        self.up_1 = UpsampleBlock(128, 128, 64)
        self.up_2 = UpsampleBlock(64, 64, 64)
        self.pred = nn.Conv2d(64, 1, kernel_size=(3,3), padding=(1,1), stride=1)

    def forward(self, logits):
        #: logits should be 4,2,384,384(MO), but we just pass one mask one time(4,1,384,384)
        logits = self.conv1(logits)
        f2 = F.relu(logits) #:4,64,192,192
        logits = self.conv2(f2)
        f4 = F.relu(logits) #:4,128,96,96
        logits = self.conv3(f4)
        f8 = F.relu(logits) #:4,256,48,48

        x = self.compress(f8) #:4,128,48,48
        x = self.up_1(f4, x) #:4,64,96,96
        x = self.up_2(f2, x) #:4,64,192,192

        x = self.pred(F.relu(x)) #:4,1,192,192
        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False) #:4,1,384,384

        return x

