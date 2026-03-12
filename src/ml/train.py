import os
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torchaudio.transforms as T
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from torch.amp import autocast, GradScaler

from src.data.data_loader import DeepWetlandsDataset
from src.ml.efficientnet_b0 import DeepWetlandsModel
from src.ml.training_logger import TrainingLogger


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

def mixup_data(x, y, alpha=0.5):
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1.0

    batch_size = x.size()[0]
    index = torch.randperm(batch_size).to(x.device)

    mixed_x = lam * x + (1 - lam) * x[index, :]
    mixed_y = lam * y + (1 - lam) * y[index, :]
    
    return mixed_x, mixed_y

def train_one_epoch(model, loader, optimizer, criterion, device, audio_transform, scaler, accumulation_steps=4):
    model.train()
    running_loss = 0.0
    optimizer.zero_grad()

    pbar = tqdm(loader, desc="Training")
    for i, (waveforms, targets) in enumerate(pbar):
        waveforms = waveforms.to(device)
        targets   = targets.to(device)

        with torch.no_grad():
            specs = audio_transform(waveforms, is_train=True)

        if np.random.rand() < 0.5:
            specs, targets = mixup_data(specs, targets, alpha=0.5)

        with autocast(device_type=device.type, dtype=torch.float16):
            outputs = model(specs)
            loss    = criterion(outputs, targets)
            loss    = loss / accumulation_steps 

        scaler.scale(loss).backward()

        if (i + 1) % accumulation_steps == 0 or (i + 1) == len(loader):
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

        step_loss = loss.item() * accumulation_steps
        running_loss += step_loss
        pbar.set_postfix(loss=step_loss)

    return running_loss / len(loader)

def validate(model, loader, criterion, device, audio_transform):
    model.eval()
    running_loss = 0.0
    all_preds   = []
    all_targets = []

    with torch.no_grad():
        pbar = tqdm(loader, desc="Validating")
        for waveforms, targets in pbar:
            waveforms = waveforms.to(device)
            targets   = targets.to(device)

            specs = audio_transform(waveforms, is_train=False)

            with autocast(device_type=device.type, dtype=torch.float16):
                outputs = model(specs)
                loss    = criterion(outputs, targets)
            
            running_loss += loss.item()
            pbar.set_postfix(loss=loss.item())

            all_preds.append(outputs.float().cpu().numpy())
            all_targets.append(targets.float().cpu().numpy())

    val_preds   = np.concatenate(all_preds,   axis=0)
    val_targets = np.concatenate(all_targets, axis=0)

    return running_loss / len(loader), val_preds, val_targets

def build_loaders(base_dir, batch_size, num_workers):
    df       = pd.read_csv(os.path.join(base_dir, "train.csv"))
    taxonomy = pd.read_csv(os.path.join(base_dir, "taxonomy.csv"))

    classes   = sorted(taxonomy['primary_label'].unique())
    label_map = {label: i for i, label in enumerate(classes)}

    counts    = df['primary_label'].value_counts()
    rare_mask = df['primary_label'].isin(counts[counts < 2].index)
    df_rare   = df[rare_mask].reset_index(drop=True)
    df_common = df[~rare_mask].reset_index(drop=True)

    print(f"Rare species (forced in training): {rare_mask.sum()} samples")

    train_df, val_df = train_test_split(
        df_common,
        test_size=0.2,
        stratify=df_common['primary_label'],
        random_state=42
    )
    train_df = pd.concat([train_df, df_rare]).reset_index(drop=True)
    val_df   = val_df.reset_index(drop=True)

    print(f"Train: {len(train_df)} samples | Val: {len(val_df)} samples")

    data_dir      = os.path.join(base_dir, "train_audio")
    train_dataset = DeepWetlandsDataset(train_df, data_dir, label_map, is_train=True)
    val_dataset   = DeepWetlandsDataset(val_df,   data_dir, label_map, is_train=False)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=True)
    val_loader   = DataLoader(val_dataset,   batch_size=batch_size, shuffle=False,
                              num_workers=num_workers, pin_memory=True)

    return train_loader, val_loader, label_map

def main():
    BASE_DIR    = "data"
    EPOCHS      = 30
    BATCH_SIZE  = 64
    ACCUM_STEPS = 1
    LR          = 1e-3
    NUM_WORKERS = 2
    RUN_NAME    = "run_003"
    
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on: {DEVICE}")

    cudnn.benchmark = True 

    train_loader, val_loader, label_map = build_loaders(BASE_DIR, BATCH_SIZE, NUM_WORKERS)
    print(f"Train batches: {len(train_loader)} | Val batches: {len(val_loader)}")

    model = DeepWetlandsModel(model_name='efficientnet_b0', num_classes=len(label_map))
    model.to(DEVICE)
    
    audio_transform = GPUAudioTransform().to(DEVICE)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.AdamW(model.parameters(), lr=LR)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
    
    scaler = GradScaler("cuda")

    logger = TrainingLogger(label_map, output_dir=f"logs/{RUN_NAME}")

    best_val_loss = float("inf")

    for epoch in range(1, EPOCHS + 1):
        print(f"\nEpoch {epoch}/{EPOCHS}")
        t0 = time.time()

        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, DEVICE, audio_transform, scaler, ACCUM_STEPS)
        val_loss, val_preds, val_targets = validate(model, val_loader, criterion, DEVICE, audio_transform)
        scheduler.step()

        epoch_time = time.time() - t0
        print(f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")

        logger.log_epoch(epoch, train_loss, val_loss, val_preds, val_targets, epoch_time)

        os.makedirs("models", exist_ok=True)
        torch.save({
            "epoch":                epoch,
            "model_state_dict":     model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "train_loss":           train_loss,
            "val_loss":             val_loss,
        }, f"models/checkpoint_epoch_{epoch}.pth")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), "models/best_model.pth")
            print(f"New best model saved! Val Loss: {val_loss:.4f}")

    logger.finalize()


if __name__ == "__main__":
    main()
