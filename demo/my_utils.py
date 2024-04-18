import torch
from torch import nn
import torch.nn.functional as F
import torchcrepe
import librosa
import numpy as np

def calc_nparam(model):
    nparam = 0
    for p in model.parameters():
        if p.requires_grad:
            nparam += p.numel()
    return nparam

def extract_pitch(signal, sr=16000, hop_length=80, fmin=100, fmax=2000, model='full', batch_size=2048, device='cuda:7'):
    audio = torch.from_numpy(signal).unsqueeze(0).float()
    
    pitch, periodicity = torchcrepe.predict(
        audio,
        sr,
        hop_length,
        fmin,
        fmax,
        model,
        batch_size=batch_size,
        device=device,
        return_periodicity=True
    )
    return pitch.flatten().numpy()

def extract_energy(signal, hop_length=80, n_fft=1024):
    spec = librosa.stft(
        signal,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=n_fft,
        center=True,
    )
    log_spec = np.log(abs(spec) + 1e-7)
    energy = np.mean(log_spec, 0)
    return energy

class ResBlock(nn.Module):
    '''
    Gaddy and Klein, 2021, https://arxiv.org/pdf/2106.01933.pdf 
    Original code:
        https://github.com/dgaddy/silent_speech/blob/master/transformer.py
    '''
    def __init__(self, num_ins, num_outs, kernel_size=3, padding=1, stride=1, dilation=1):
        super().__init__()

        self.conv1 = nn.Conv1d(num_ins, num_outs, kernel_size, padding=(kernel_size-1)//2*dilation, stride=stride, dilation=dilation)
        self.bn1 = nn.BatchNorm1d(num_outs)
        self.conv2 = nn.Conv1d(num_outs, num_outs, kernel_size, padding=(kernel_size-1)//2*dilation, stride=stride, dilation=dilation)
        self.bn2 = nn.BatchNorm1d(num_outs)

        if stride != 1 or num_ins != num_outs:
            self.residual_path = nn.Conv1d(num_ins, num_outs, 1, stride=stride)
            self.res_norm = nn.BatchNorm1d(num_outs)
        else:
            self.residual_path = None

    def forward(self, x):
        input_value = x

        x = F.relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))

        if self.residual_path is not None:
            res = self.res_norm(self.residual_path(input_value))
        else:
            res = input_value

        return F.relu(x + res)

class DilatedConvStack(nn.Module):
    def __init__(self, hidden_dim, kernel_size, stride, dilations):
        super().__init__()
        self.stack = []
        for dilation in dilations:
            self.stack.append(ResBlock(num_ins=hidden_dim, num_outs=hidden_dim, kernel_size=kernel_size, stride=stride, dilation=dilation))
        self.stack = nn.Sequential(*self.stack)
        
    def forward(self, x):
        return self.stack(x)

class DilatedConvEncoder(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, dilations, nstacks):
        super().__init__()
        self.in_conv = nn.Conv1d(in_channels, out_channels, kernel_size, stride, padding=(kernel_size-1)//2)
        self.stacks = []
        for _ in range(nstacks):
            self.stacks.append(DilatedConvStack(hidden_dim=out_channels, kernel_size=kernel_size, stride=stride, dilations=dilations))
        self.stacks = nn.Sequential(*self.stacks)
        self.out_conv = nn.Conv1d(out_channels, out_channels, kernel_size, stride, padding=(kernel_size-1)//2)
        
    def forward(self, x):
        x = self.in_conv(x)
        x = self.stacks(x)
        out = self.out_conv(x)
        return out
