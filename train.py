# train.py
# Ví dụ:
#   python train.py --model simple
#   python train.py --model unet
#   python train.py --model transfer

import argparse
import time
import torch
import torch.nn as nn
from pathlib import Path

from config import (
    SEED, SIMPLE_EPOCHS, UNET_EPOCHS, TRANSFER_EPOCHS,
    FREEZE_EPOCHS, LR_SCRATCH, LR_HEAD, LR_FINETUNE, OUTPUT_DIR
)
from dataset import make_dataloaders
from models import build_model, get_logits, set_backbone_trainable, count_parameters
from utils import set_seed, confusion_counts, metrics_from_counts, ensure_dirs


def validate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    tp = fp = fn = tn = 0

    with torch.no_grad():
        for images, masks in loader:
            images = images.to(device)
            masks = masks.to(device)

            logits = get_logits(model(images))
            loss = criterion(logits, masks)
            total_loss += loss.item() * images.size(0)

            a, b, c, d = confusion_counts(logits, masks)
            tp += a; fp += b; fn += c; tn += d

    metrics = metrics_from_counts(tp, fp, fn, tn)
    return total_loss / len(loader.dataset), metrics


def train_one_model(model_name):
    set_seed(SEED)
    ensure_dirs()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    train_loader, val_loader, _ = make_dataloaders(download=True)

    # Transfer model cần pretrained weights lúc bắt đầu training.
    model = build_model(model_name, pretrained_transfer=(model_name == "transfer"))
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()

    if model_name == "simple":
        epochs = SIMPLE_EPOCHS
        optimizer = torch.optim.Adam(model.parameters(), lr=LR_SCRATCH)
    elif model_name == "unet":
        epochs = UNET_EPOCHS
        optimizer = torch.optim.Adam(model.parameters(), lr=LR_SCRATCH)
    else:
        epochs = TRANSFER_EPOCHS

        # Giai đoạn đầu: freeze backbone, chỉ train classifier.
        set_backbone_trainable(model, False)
        optimizer = torch.optim.Adam(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=LR_HEAD,
        )

    total_params, trainable_params = count_parameters(model)
    print(f"Total params: {total_params:,}")
    print(f"Trainable params initially: {trainable_params:,}")

    best_iou = -1.0
    best_path = Path(OUTPUT_DIR, "checkpoints", f"{model_name}_best.pt")
    start_time = time.time()

    for epoch in range(epochs):
        # Sau FREEZE_EPOCHS, unfreeze transfer backbone và fine-tune LR nhỏ.
        if model_name == "transfer" and epoch == FREEZE_EPOCHS:
            print("\nUnfreezing backbone -> starting full-model fine-tuning")
            set_backbone_trainable(model, True)
            optimizer = torch.optim.Adam(model.parameters(), lr=LR_FINETUNE)

        model.train()
        # Disabling gradients alone does not stop BatchNorm running-stat updates.
        if model_name == "transfer" and epoch < FREEZE_EPOCHS:
            model.backbone.eval()
        running_loss = 0.0
        seen_samples = 0

        for images, masks in train_loader:
            images = images.to(device)
            masks = masks.to(device)

            optimizer.zero_grad()
            logits = get_logits(model(images))
            loss = criterion(logits, masks)
            loss.backward()
            optimizer.step()

            batch_size = images.size(0)
            running_loss += loss.item() * batch_size
            seen_samples += batch_size

        train_loss = running_loss / seen_samples
        val_loss, val_metrics = validate(model, val_loader, criterion, device)

        print(
            f"Epoch {epoch+1:02d}/{epochs} | "
            f"train_loss={train_loss:.4f} | "
            f"val_loss={val_loss:.4f} | "
            f"IoU={val_metrics['iou']:.4f} | "
            f"Dice={val_metrics['dice']:.4f}"
        )

        # Chọn checkpoint bằng validation IoU.
        if val_metrics["iou"] > best_iou:
            best_iou = val_metrics["iou"]
            torch.save(model.state_dict(), best_path)
            print("  -> saved best checkpoint")

    elapsed = time.time() - start_time
    print(f"\nDone. Best val IoU = {best_iou:.4f}")
    print(f"Checkpoint: {best_path}")
    print(f"Training time: {elapsed/60:.1f} minutes")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        required=True,
        choices=["simple", "unet", "transfer"],
    )
    args = parser.parse_args()
    train_one_model(args.model)
