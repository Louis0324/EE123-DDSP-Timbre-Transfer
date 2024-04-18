import torch
import torch.nn as nn
from encoder import EncoderDilatedConvSinCos
from harmonic import HarmonicOscillatorSinCos
from noise import FilteredNoiseGenerator

class Vocoder(nn.Module):
    def __init__(self, hidden_dim, nharmonics, nbands, attenuate=0.02, fs=16000, framesize=80, temperature=1., dilations=[1, 2, 4, 8, 16], nstacks=1, use_harmonic_conv=False, reverb_len=16001):
        super().__init__()
        self.encoder = EncoderDilatedConvSinCos(hidden_dim, nharmonics, nbands, attenuate, dilations, nstacks)
        self.harmonic = HarmonicOscillatorSinCos(fs, framesize, temperature, use_harmonic_conv)
        self.noise = FilteredNoiseGenerator(framesize)
        self.reverb = nn.Conv1d(in_channels=1, out_channels=1, kernel_size=reverb_len, padding=(reverb_len-1)//2, bias=False)            

    def forward(self, f0, loudness):
        """
        #### input ####
        f0          : in shape [B, 1, t], calculated at f_model 
        loudness    : in shape [B, 1, t], calculated at f_model   
        
        #### output ####
        audio      : in shape [B, t*framesize]
        
        """
        # going through encoder to get the control signals
        cn_sin, cn_cos, an_sin, an_cos, H = self.encoder(f0, loudness)
        
        # generate harmonic components
        harmonics = self.harmonic(f0, cn_sin, cn_cos, an_sin, an_cos) # [B, t*framesize]
            
        # generate filtered noise
        noise = self.noise(H) # [B, t*framesize]
        
        # additive synthesis
        audio = harmonics + noise # [B, t*framesize]
        
        # reverb
        audio = self.reverb(audio.unsqueeze(1)).squeeze(1)
        
        return audio
        

    
    

        
        
