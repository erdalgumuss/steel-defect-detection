import torch
from tqdm import tqdm

def train_one_epoch(model, loader, optimizer, criterion, device, scaler=None):
    model.train()
    total_loss = 0
    for imgs, masks in tqdm(loader, desc="Train", leave=False):
        imgs, masks = imgs.to(device), masks.to(device).float()
        optimizer.zero_grad()

        with torch.cuda.amp.autocast(enabled=scaler is not None):
            logits = model(imgs)
            loss = criterion(logits, masks)

        if scaler:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        total_loss += loss.item()
    return total_loss / len(loader)

def validate_one_epoch(model, loader, criterion, metric_fn, device):
    model.eval()
    total_loss = 0
    total_metric = 0
    with torch.no_grad():
        for imgs, masks in tqdm(loader, desc="Val", leave=False):
            imgs, masks = imgs.to(device), masks.to(device).float()
            logits = model(imgs)
            loss = criterion(logits, masks)
            metric = metric_fn(logits, masks)

            total_loss += loss.item()
            total_metric += metric
    return total_loss / len(loader), total_metric / len(loader)

def fit(model, train_loader, val_loader, optimizer, criterion, metric_fn,
        device, epochs=10, scheduler=None, use_amp=True, save_path="outputs/checkpoints/best_model.pth"):
    
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
    best_metric = -1.0

    for epoch in range(1, epochs + 1):
        print(f"\nEpoch {epoch}/{epochs}")
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device, scaler)
        val_loss, val_metric = validate_one_epoch(model, val_loader, criterion, metric_fn, device)

        if scheduler:
            scheduler.step()

        print(f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val Dice: {val_metric:.4f}")

        # Save best model
        if val_metric > best_metric:
            best_metric = val_metric
            torch.save(model.state_dict(), save_path)
            print(f"[INFO] Saved best model with Dice {best_metric:.4f}")
