"""
eval_network.py - Evaluation version of the network
The logic is basically the same
but with top-k and some implementation optimization

The trailing number of a variable usually denote the stride
e.g. f16 -> encoded features with stride 16
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from model.modules import *
from model.network import Decoder
from model.network import SCM


class STCN(nn.Module):
    def __init__(self):
        super().__init__()
        self.key_encoder = KeyEncoder() 
        self.value_encoder = ValueEncoder() 

        # Projection from f16 feature space to key space
        self.key_proj = KeyProjection(1024, keydim=64)

        # Compress f16 a bit to use in decoding later on
        self.key_comp = nn.Conv2d(1024, 512, kernel_size=3, padding=1)

        self.scm = SCM()
        self.decoder = Decoder()

    def encode_value(self, frame, kf16, masks): 
        k, _, h, w = masks.shape

        # Extract memory key/value for a frame with multiple masks
        frame = frame.view(1, 3, h, w).repeat(k, 1, 1, 1)
        # Compute the "others" mask
        if k != 1:
            others = torch.cat([
                torch.sum(
                    masks[[j for j in range(k) if i!=j]]
                , dim=0, keepdim=True)
            for i in range(k)], 0)
        else:
            others = torch.zeros_like(masks)

        f16 = self.value_encoder(frame, kf16.repeat(k,1,1,1), masks, others)

        return f16.unsqueeze(2)

    def encode_key(self, frame):
        f16, f8, f4 = self.key_encoder(frame)
        k16 = self.key_proj(f16)
        f16_thin = self.key_comp(f16)

        return k16, f16_thin, f16, f8, f4

    def segment_with_query(self, mem_bank, qf8, qf4, qk16, qv16, prev_mask):
        k = mem_bank.num_objects       #: prev_mask: MO:2,1,480,912    SO:1,1,480,864

        readout_mem = mem_bank.match_memory(qk16) #: MO:2,512,30,57    SO:1,512,30,54
        qv16 = qv16.expand(k, -1, -1, -1)         #: MO:2,512,30,57    SO:1,512,30,54
        qv16 = torch.cat([readout_mem, qv16], 1)  #: MO:2,1024,30,57   SO:1,1024,30,54
        qv16 = self.scm(qv16, prev_mask)
        """
        if k == 1:
            qv16 = self.scm(qv16, prev_mask)
        if k == 2:
            tmp1 = self.scm(qv16[0].unsqueeze(0), prev_mask[0].unsqueeze(0))
            tmp2 = self.scm(qv16[1].unsqueeze(0), prev_mask[1].unsqueeze(0))
            qv16 = torch.cat([tmp1, tmp2], 0)
        """

        return torch.sigmoid(self.decoder(qv16, qf8, qf4))
