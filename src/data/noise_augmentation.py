import torch
import torch.nn as nn
import math

class EnvironmentalNoiseAugmentation(nn.Module):
    def __init__(self, p=0.5, min_snr_db=3.0, max_snr_db=15.0):
        super().__init__()
        self.p = p
        self.min_snr_db = min_snr_db
        self.max_snr_db = max_snr_db

    def _generate_colored_noise(self, shape, device, color='pink'):
        white_noise = torch.randn(shape, device=device)
        
        if color == 'white':
            return white_noise

        fft = torch.fft.rfft(white_noise)
        freqs = torch.fft.rfftfreq(shape[-1], d=1.0, device=device)

        freqs[0] = 1.0 
        
        if color == 'pink':
            fft = fft / torch.sqrt(freqs)
        elif color == 'brown':
            fft = fft / freqs
            
        noise = torch.fft.irfft(fft, n=shape[-1])
        return noise

    def forward(self, waveforms):
        if not self.training or torch.rand(1).item() > self.p:
            return waveforms

        batch_size = waveforms.shape[0]
        device = waveforms.device

        color = torch.rand(1).item()
        if color < 0.33:
            noise_type = 'white'
        elif color < 0.66:
            noise_type = 'pink'
        else:
            noise_type = 'brown'

        noise = self._generate_colored_noise(waveforms.shape, device, color=noise_type)

        eps = 1e-8
        rms_signal = torch.sqrt(torch.mean(waveforms ** 2, dim=-1, keepdim=True) + eps)
        rms_noise = torch.sqrt(torch.mean(noise ** 2, dim=-1, keepdim=True) + eps)

        target_snr = torch.empty((batch_size,) + (1,) * (waveforms.ndim - 1), device=device).uniform_(self.min_snr_db, self.max_snr_db)

        target_rms_noise = rms_signal / (10 ** (target_snr / 20.0))

        scaled_noise = noise * (target_rms_noise / rms_noise)

        noisy_waveforms = waveforms + scaled_noise

        return torch.clamp(noisy_waveforms, min=-1.0, max=1.0)
