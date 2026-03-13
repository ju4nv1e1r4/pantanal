import os
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from torch.amp import autocast, GradScaler

from src.data.data_loader import DeepWetlandsDataset
from src.ml.model import DeepWetlandsModel
from src.ml.training_logger import TrainingLogger
from src.ml.audio_transform import GPUAudioTransform


def mixup_data(x: torch.Tensor, y: torch.Tensor, alpha: float = 0.2):
    if alpha > 0:
        lam = float(np.random.beta(alpha, alpha))
    else:
        lam = 1.0

    batch_size = x.size(0)
    index = torch.randperm(batch_size, device=x.device)

    mixed_x = lam * x + (1 - lam) * x[index]
    mixed_y = lam * y + (1 - lam) * y[index]

    return mixed_x, mixed_y

def train_one_epoch(
    model, loader, optimizer, criterion, device,
    audio_transform, scaler, accumulation_steps=2
):
    model.train()
    running_loss = 0.0
    optimizer.zero_grad()

    pbar = tqdm(loader, desc="Training")
    for i, (waveforms, targets) in enumerate(pbar):
        waveforms = waveforms.to(device)
        targets   = targets.to(device)

        if np.random.rand() < 0.5:
            waveforms, targets = mixup_data(waveforms, targets, alpha=0.2)

        with torch.no_grad():
            specs = audio_transform(waveforms, is_train=True)

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
        pbar.set_postfix(loss=f"{step_loss:.4f}")

    return running_loss / len(loader)

def validate(model, loader, criterion, device, audio_transform):
    model.eval()
    running_loss = 0.0
    all_preds    = []
    all_targets  = []

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
            pbar.set_postfix(loss=f"{loss.item():.4f}")

            all_preds.append(outputs.float().cpu().numpy())
            all_targets.append(targets.float().cpu().numpy())

    val_preds   = np.concatenate(all_preds,   axis=0)
    val_targets = np.concatenate(all_targets, axis=0)

    return running_loss / len(loader), val_preds, val_targets

def build_loaders(base_dir: str, batch_size: int, num_workers: int):
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
        random_state=42,
    )
    train_df = pd.concat([train_df, df_rare]).reset_index(drop=True)
    val_df   = val_df.reset_index(drop=True)

    print(f"Train: {len(train_df)} samples | Val: {len(val_df)} samples")

    data_dir      = os.path.join(base_dir, "train_audio")
    train_dataset = DeepWetlandsDataset(train_df, data_dir, label_map, is_train=True)
    val_dataset   = DeepWetlandsDataset(val_df,   data_dir, label_map, is_train=False)

    class_counts  = train_df['primary_label'].value_counts().to_dict()
    sample_weights = train_df['primary_label'].map(
        lambda lbl: 1.0 / class_counts[lbl]
    ).values
    sampler = torch.utils.data.WeightedRandomSampler(
        weights=torch.tensor(sample_weights, dtype=torch.float),
        num_samples=len(train_dataset),
        replacement=True,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    return train_loader, val_loader, label_map

def main():
    BASE_DIR    = "data"
    EPOCHS      = 30
    BATCH_SIZE  = 64
    ACCUM_STEPS = 1
    LR          = 2e-3
    NUM_WORKERS = 4
    RUN_NAME    = "run_005"
    PATIENCE    = 15

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on: {DEVICE}")

    cudnn.benchmark = True

    train_loader, val_loader, label_map = build_loaders(BASE_DIR, BATCH_SIZE, NUM_WORKERS)
    print(f"Train batches: {len(train_loader)} | Val batches: {len(val_loader)}")

    model = DeepWetlandsModel(model_name='efficientnet_b0', num_classes=len(label_map))
    model.to(DEVICE)

    audio_transform = GPUAudioTransform().to(DEVICE)

    criterion = nn.BCEWithLogitsLoss()

    backbone_params = [p for n, p in model.named_parameters() if 'head' not in n]
    head_params     = [p for n, p in model.named_parameters() if 'head'     in n]

    optimizer = optim.AdamW([
        {'params': backbone_params, 'lr': LR * 0.1},   # 1e-4
        {'params': head_params,     'lr': LR},         # 1e-3
    ], weight_decay=1e-4)

    WARMUP_EPOCHS = 3

    def lr_lambda(epoch):
        if epoch < WARMUP_EPOCHS:
            return (epoch + 1) / WARMUP_EPOCHS   # linear warmup
        progress = (epoch - WARMUP_EPOCHS) / max(1, EPOCHS - WARMUP_EPOCHS)
        return 0.5 * (1.0 + np.cos(np.pi * progress))   # cosine decay

    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)

    scaler = GradScaler("cuda")

    logger = TrainingLogger(
        model_name="efficientnet_b0",
        label_map=label_map,
        output_dir=f"logs/{RUN_NAME}",
    )

    os.makedirs("models", exist_ok=True)

    best_auc      = 0.0
    epochs_no_gain = 0

    for epoch in range(1, EPOCHS + 1):
        print(f"\nEpoch {epoch}/{EPOCHS}  |  lr_backbone={optimizer.param_groups[0]['lr']:.2e}  lr_head={optimizer.param_groups[1]['lr']:.2e}")
        t0 = time.time()

        train_loss = train_one_epoch(
            model, train_loader, optimizer, criterion,
            DEVICE, audio_transform, scaler, ACCUM_STEPS,
        )
        val_loss, val_preds, val_targets = validate(
            model, val_loader, criterion, DEVICE, audio_transform,
        )
        scheduler.step()

        epoch_time = time.time() - t0
        print(f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Time: {epoch_time:.0f}s")

        logger.log_epoch(epoch, train_loss, val_loss, val_preds, val_targets, epoch_time)

        from sklearn.metrics import roc_auc_score
        import warnings
        warnings.filterwarnings('ignore')
        try:
            macro_auc = roc_auc_score(val_targets, val_preds, average='macro')
        except ValueError:
            macro_auc = 0.0

        torch.save({
            "epoch":                epoch,
            "model_state_dict":     model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "train_loss":           train_loss,
            "val_loss":             val_loss,
            "macro_auc":            macro_auc,
        }, f"models/checkpoint_epoch_{epoch:02d}.pth")

        if macro_auc > best_auc:
            best_auc = macro_auc
            epochs_no_gain = 0
            torch.save(model.state_dict(), "models/best_model.pth")
            print(f"  ✦ New best model — macro-AUC: {best_auc:.4f}")
        else:
            epochs_no_gain += 1
            print(f"  No gain for {epochs_no_gain}/{PATIENCE} epochs (best AUC: {best_auc:.4f})")

        if epochs_no_gain >= PATIENCE:
            print(f"\nEarly stopping triggered after {epoch} epochs.")
            break

    logger.finalize()
    print(f"\nTraining complete. Best macro-AUC: {best_auc:.4f}")


if __name__ == "__main__":
    main()
