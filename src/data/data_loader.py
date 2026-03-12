import os
import pandas as pd
import numpy as np
import torch
import torchaudio
import torchaudio.transforms as T
from torch.utils.data import Dataset, DataLoader
import soundfile as sf


class PantanalDataset(Dataset):
    def __init__(self, df, data_dir, label_map, target_sample_rate=32000, duration=5, is_train=True):
        """
        Dataset for the BirdCLEF+ Pantanal 2026 competition.
        
        Args:
            df: DataFrame with metadata (train.csv).
            data_dir: Path to the 'train_audio' folder.
            label_map: Dictionary mapping 'primary_label' to index (0-233).
            target_sample_rate: Competition fixed sampling rate (32kHz).
            duration: Duration in seconds for each sample.
            is_train: Defines if data augmentation is applied.
        """
        self.df = df
        self.data_dir = data_dir
        self.label_map = label_map
        self.target_sample_rate = target_sample_rate
        self.duration = duration
        self.num_samples = target_sample_rate * duration
        self.is_train = is_train

        # Fixed transformations: MelSpectrogram
        self.mel_spectrogram = T.MelSpectrogram(
            sample_rate=self.target_sample_rate,
            n_fft=2048,
            hop_length=512,
            n_mels=128,
            f_min=0,
            f_max=16000
        )
        self.db_transform = T.AmplitudeToDB()

        # Augmentations via Torchaudio (SpecAugment)
        self.time_mask = T.TimeMasking(time_mask_param=30)
        self.freq_mask = T.FrequencyMasking(freq_mask_param=15)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        file_path = os.path.join(self.data_dir, row['filename'])

        try:
            data, sample_rate = sf.read(file_path, dtype='float32')
            waveform = torch.tensor(data).T
            if waveform.ndim == 1:
                waveform = waveform.unsqueeze(0)
        except Exception as e:
            print(f"Error loading file: {file_path}")
            raise e

        if sample_rate != self.target_sample_rate:
            resampler = T.Resample(sample_rate, self.target_sample_rate)
            waveform = resampler(waveform)

        if waveform.shape[1] > self.num_samples:
            if self.is_train:
                start = np.random.randint(0, waveform.shape[1] - self.num_samples)
            else:
                start = 0
            waveform = waveform[:, start:start + self.num_samples]
        else:
            padding = self.num_samples - waveform.shape[1]
            waveform = torch.nn.functional.pad(waveform, (0, padding))

        spec = self.mel_spectrogram(waveform)
        spec = self.db_transform(spec)

        if self.is_train:
            spec = self.freq_mask(spec)
            spec = self.time_mask(spec)

        label_idx = self.label_map[row['primary_label']]
        target = torch.zeros(len(self.label_map))
        target[label_idx] = 1.0

        return spec, target

def get_dataloader(csv_path, taxonomy_path, data_dir, batch_size=32, is_train=True, num_workers=4):
    df = pd.read_csv(csv_path)
    taxonomy = pd.read_csv(taxonomy_path)

    classes = sorted(taxonomy['primary_label'].unique())
    label_map = {label: i for i, label in enumerate(classes)}
    
    dataset = PantanalDataset(df, data_dir, label_map, is_train=is_train)
    
    loader = DataLoader(
        dataset, 
        batch_size=batch_size, 
        shuffle=is_train, 
        num_workers=num_workers,
        pin_memory=True
    )
    
    return loader, label_map

if __name__ == "__main__":
    BASE_DIR = "/home/juan/x/pantanal/data" or "../data"
    
    loader, l_map = get_dataloader(
        csv_path=os.path.join(BASE_DIR, "train.csv"),
        taxonomy_path=os.path.join(BASE_DIR, "taxonomy.csv"),
        data_dir=os.path.join(BASE_DIR, "train_audio"),
        batch_size=16,
        num_workers=0
    )

    specs, targets = next(iter(loader))
    print(f"Shape of the Batch of Spectrograms: {specs.shape}") # [Batch, Canais, Freq, Tempo]
    print(f"Shape of Batch of Labels: {targets.shape}")      # [Batch, 234]
    print("DataLoader's done!")
