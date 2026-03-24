import os

import numpy as np
import pandas as pd
import soundfile as sf
import torch
import torchaudio.functional as F_audio
from torch.utils.data import DataLoader, Dataset


class DeepWetlandsDataset(Dataset):
    def __init__(
        self,
        df,
        data_dir,
        label_map,
        target_sample_rate=32000,
        duration=5,
        is_train=True,
    ):
        self.df = df
        self.data_dir = data_dir
        self.label_map = label_map
        self.target_sample_rate = target_sample_rate
        self.duration = duration
        self.num_samples = target_sample_rate * duration
        self.is_train = is_train

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        file_path = os.path.join(self.data_dir, row["filename"])

        try:
            data, sample_rate = sf.read(file_path, dtype="float32")
            waveform = torch.tensor(data.copy())

            if waveform.ndim == 1:
                waveform = waveform.unsqueeze(0)  # Mono: [N] -> [1, N]
            else:
                waveform = waveform.permute(1, 0)  # Stereo: [N, C] -> [C, N]
                waveform = waveform.mean(dim=0, keepdim=True)

        except Exception as e:
            print(f"Error loading file: {file_path}")
            waveform = torch.zeros((1, self.num_samples))
            sample_rate = self.target_sample_rate

        if sample_rate != self.target_sample_rate:
            waveform = F_audio.resample(waveform, sample_rate, self.target_sample_rate)

        num_samples_audio = waveform.shape[1]

        if num_samples_audio < self.num_samples:
            padding = self.num_samples - num_samples_audio
            waveform = torch.nn.functional.pad(waveform, (0, padding))
        elif num_samples_audio > self.num_samples * 6:
            windows = waveform.unfold(1, self.num_samples, self.num_samples // 2)
            energies = torch.sum(windows**2, dim=2)
            best_idx = torch.argmax(energies)
            waveform = windows[0, best_idx].unsqueeze(0)
        elif num_samples_audio > self.num_samples:
            if self.is_train:
                start = np.random.randint(0, num_samples_audio - self.num_samples)
            else:
                start = (num_samples_audio - self.num_samples) // 2
            waveform = waveform[:, start : start + self.num_samples]

        label_idx = self.label_map[row["primary_label"]]
        target = torch.zeros(len(self.label_map))
        target[label_idx] = 1.0

        return waveform, target


def get_dataloader(
    csv_path, taxonomy_path, data_dir, batch_size=32, is_train=True, num_workers=4
):
    df = pd.read_csv(csv_path)
    taxonomy = pd.read_csv(taxonomy_path)

    classes = sorted(taxonomy["primary_label"].unique())
    label_map = {label: i for i, label in enumerate(classes)}

    dataset = DeepWetlandsDataset(df, data_dir, label_map, is_train=is_train)

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=is_train,
        num_workers=num_workers,
        pin_memory=True,
    )

    return loader, label_map
