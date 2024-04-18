import torch
from torch import nn
import math
from my_utils import DilatedConvEncoder

class EncoderDilatedConvSinCos(nn.Module):
    def __init__(self, hidden_dim=256, nharmonics=50, nbands=65, attenuate=0.02, dilations=[1, 2, 4, 8, 16], nstacks=4):
        super().__init__()        
        self.attenuate = attenuate 
        self.nharmonics = nharmonics
        self.convencoder = DilatedConvEncoder(in_channels=2, out_channels=hidden_dim, kernel_size=3, stride=1, dilations=dilations, nstacks=nstacks)
        
        self.head_amp = nn.Sequential(
            nn.Linear(hidden_dim, (nharmonics+1)*2)
        )
        
        self.head_H = nn.Sequential(
            nn.Linear(hidden_dim, nbands),  
            nn.LayerNorm(nbands)
        )

    # Scale sigmoid as per original DDSP paper
    def _scaled_sigmoid(self, x):
        return 2.0 * (torch.sigmoid(x) ** math.log(10)) + 1e-7

    def forward(self, f0, loudness):
        """
        #### input ####
        f0          : in shape [B, 1, t], calculated at f_model 
        loudness    : in shape [B, 1, t], calculated at f_model
        
        #### output ####
        cn          : in shape [B, nharmonics, t], calculated at f_model
        an          : in shape [B, 1, t], calculated at f_model 
        H           : in shape [B, t, nbands], calculated at f_model
        
        """
        # normalize f0 to be within [0, 1]
        f0 = f0 / 2000
        
        # conv encoding
        in_feat = torch.concat([f0, loudness], dim=1) # [B, 2, t]
        out_feat = self.convencoder(in_feat).transpose(1,2) # [B, t, hidden_dim]
        
        # output heads
        amp = self.head_amp(out_feat) # [B, t, (nharmonics+1)*2]
        amp_sin = amp[:,:,:(self.nharmonics+1)] # [B, t, nharmonics+1]
        amp_cos = amp[:,:,(self.nharmonics+1):] # [B, t, nharmonics+1]
        
        cn_sin = (amp_sin[:, :, 1:]).transpose(1,2) # [B, nharmonics, t]
        an_sin = self._scaled_sigmoid(amp_sin[:, :, 0].unsqueeze(-1)).transpose(1,2)
        
        cn_cos = (amp_cos[:, :, 1:]).transpose(1,2) # [B, nharmonics, t]
        an_cos = self._scaled_sigmoid(amp_cos[:, :, 0].unsqueeze(-1)).transpose(1,2)
        
        H  = self._scaled_sigmoid(self.head_H(out_feat))*self.attenuate

        return cn_sin, cn_cos, an_sin, an_cos, H
