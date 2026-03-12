import os
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

from src.data.data_loader import DeepWetlandsDataset
from src.ml.efficientnet_b0 import DeepWetlandsModel
from src.ml.training_logger import TrainingLogger


def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    running_loss = 0.0

    pbar = tqdm(loader, desc="Training")
    for specs, targets in pbar:
        specs   = specs.to(device)
        targets = targets.to(device)

        optimizer.zero_grad()
        outputs = model(specs)
        loss    = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        pbar.set_postfix(loss=loss.item())

    return running_loss / len(loader)

def validate(model, loader, criterion, device):
    """Returns loss avg + arrays of pred/targets to logger."""
    model.eval()
    running_loss = 0.0
    all_preds   = []
    all_targets = []

    with torch.no_grad():
        pbar = tqdm(loader, desc="Validating")
        for specs, targets in pbar:
            specs   = specs.to(device)
            targets = targets.to(device)

            outputs = model(specs)
            loss    = criterion(outputs, targets)
            running_loss += loss.item()
            pbar.set_postfix(loss=loss.item())

            all_preds.append(outputs.cpu().numpy())
            all_targets.append(targets.cpu().numpy())

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
    EPOCHS      = 20
    BATCH_SIZE  = 32
    LR          = 1e-3
    NUM_WORKERS = 4
    RUN_NAME    = "run_001"
    DEVICE      = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on: {DEVICE}")

    train_loader, val_loader, label_map = build_loaders(BASE_DIR, BATCH_SIZE, NUM_WORKERS)
    print(f"Train batches: {len(train_loader)} | Val batches: {len(val_loader)}")

    model     = DeepWetlandsModel(model_name='efficientnet_b0', num_classes=len(label_map))
    model.to(DEVICE)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.AdamW(model.parameters(), lr=LR)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    logger = TrainingLogger(label_map, output_dir=f"logs/{RUN_NAME}")

    best_val_loss = float("inf")

    for epoch in range(1, EPOCHS + 1):
        print(f"\nEpoch {epoch}/{EPOCHS}")
        t0 = time.time()

        train_loss                        = train_one_epoch(model, train_loader, optimizer, criterion, DEVICE)
        val_loss, val_preds, val_targets  = validate(model, val_loader, criterion, DEVICE)
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
            print(f"  → New best model saved! Val Loss: {val_loss:.4f}")

    logger.finalize()


if __name__ == "__main__":
    main()
