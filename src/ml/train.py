import os
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from src.data.data_loader import get_dataloader
from src.ml.efficientnet_b0 import PantanalPulseModel

def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    running_loss = 0.0
    
    pbar = tqdm(loader, desc="Training")
    for specs, targets in pbar:
        specs = specs.to(device)
        targets = targets.to(device)
        
        optimizer.zero_grad()
        
        # Forward pass
        outputs = model(specs)
        loss = criterion(outputs, targets)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        pbar.set_postfix(loss=loss.item())
        
    return running_loss / len(loader)

def validate(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    # here I can even add the Macro-AUC calculation in the future...
    
    with torch.no_grad():
        for specs, targets in loader:
            specs = specs.to(device)
            targets = targets.to(device)
            
            outputs = model(specs)
            loss = criterion(outputs, targets)
            running_loss += loss.item()
            
    return running_loss / len(loader)

def main():
    BASE_DIR = "data" or "../data"
    EPOCHS = 10
    BATCH_SIZE = 32
    LR = 1e-3
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on: {DEVICE}")

    train_loader, label_map = get_dataloader(
        csv_path=os.path.join(BASE_DIR, "train.csv"),
        taxonomy_path=os.path.join(BASE_DIR, "taxonomy.csv"),
        data_dir=os.path.join(BASE_DIR, "train_audio"),
        batch_size=BATCH_SIZE,
        is_train=True,
        num_workers=4
    )

    model = PantanalPulseModel(model_name='efficientnet_b0', num_classes=len(label_map))
    model.to(DEVICE)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.AdamW(model.parameters(), lr=LR)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    for epoch in range(EPOCHS):
        print(f"\nEpoch {epoch+1}/{EPOCHS}")
        
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, DEVICE)
        scheduler.step()
        
        print(f"Loss de Treino: {train_loss:.4f}")
        
        # checkpoint
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': train_loss,
        }, f"models/checkpoint_epoch_{epoch+1}.pth")

if __name__ == "__main__":
    os.makedirs("models", exist_ok=True)
    main()
