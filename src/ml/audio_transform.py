import torchaudio.transforms as T
import torch.nn as nn

class GPUAudioTransform(nn.Module):
    def __init__(self, target_sample_rate=32000):
        super().__init__()
        self.mel_spectrogram = T.MelSpectrogram(
            sample_rate=target_sample_rate,
            n_fft=2048,
            hop_length=512,
            n_mels=128,
            f_min=0,
            f_max=16000
        )
        self.db_transform = T.AmplitudeToDB()
        self.time_mask = T.TimeMasking(time_mask_param=30)
        self.freq_mask = T.FrequencyMasking(freq_mask_param=15)

    def forward(self, waveform, is_train=False):
        spec = self.mel_spectrogram(waveform)
        spec = self.db_transform(spec)
        if is_train:
            spec = self.freq_mask(spec)
            spec = self.time_mask(spec)
        return spec
