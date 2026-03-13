import torch
import torch.nn as nn
import torchaudio.transforms as T


class GPUAudioTransform(nn.Module):
    """
    GPU-accelerated mel spectrogram + augmentation pipeline.
    Ajustado cirurgicamente para Amphibians e Insecta.
    """
    def __init__(
        self,
        target_sample_rate: int = 32000,
        n_fft: int = 2048,
        hop_length: int = 512,
        n_mels: int = 224,      # NOTE: increaed from 128 to 224. More resolution to insects noise.
        f_min: float = 50.0,    # NOTE: from 500 para 50. To capture low amphibian croak.
        f_max: float = 16000.0,
        time_mask_param: int = 30,
        freq_mask_param: int = 15,
        n_time_masks: int = 2,
        n_freq_masks: int = 2,
    ):
        super().__init__()

        self.mel_spectrogram = T.MelSpectrogram(
            sample_rate=target_sample_rate,
            n_fft=n_fft,
            hop_length=hop_length,
            n_mels=n_mels,
            f_min=f_min,
            f_max=f_max,
        )
        self.db_transform = T.AmplitudeToDB()

        self.time_masks = nn.ModuleList([
            T.TimeMasking(time_mask_param=time_mask_param, iid_masks=True)
            for _ in range(n_time_masks)
        ])
        self.freq_masks = nn.ModuleList([
            T.FrequencyMasking(freq_mask_param=freq_mask_param, iid_masks=True)
            for _ in range(n_freq_masks)
        ])

    def forward(self, waveform: torch.Tensor, is_train: bool = False) -> torch.Tensor:
        spec = self.mel_spectrogram(waveform)   # [B, 1, n_mels, T]
        spec = self.db_transform(spec)

        if is_train:
            for fm in self.freq_masks:
                spec = fm(spec)
            for tm in self.time_masks:
                spec = tm(spec)

        return spec
