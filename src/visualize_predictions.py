# visualize_predictions.py
"""Lấy ví dụ output segmentation của 3 model trên cùng vài ảnh test.

Ví dụ:
    python visualize_predictions.py --num-samples 4
    python visualize_predictions.py --num-samples 6 --split val
"""
import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image

from config import OUTPUT_DIR
from dataset import make_dataloaders
from models import build_model, get_logits

MODEL_NAMES = ("simple", "unet", "transfer")

# Cùng MEAN/STD với dataset.py, dùng để hoàn tác normalize khi hiển thị.
MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)


def denormalize(image):
    return (image * STD + MEAN).clamp(0, 1)


def load_trained_model(name, device):
    """Load đúng cách evaluate.py đang load: không tải pretrained lại."""
    checkpoint = Path(OUTPUT_DIR, "checkpoints", f"{name}_best.pt")
    if not checkpoint.exists():
        raise FileNotFoundError(
            f"Thiếu checkpoint của '{name}': {checkpoint}. "
            f"Chạy `python train.py --model {name}` trước."
        )
    model = build_model(name, pretrained_transfer=False).to(device)
    state = torch.load(checkpoint, map_location=device, weights_only=True)
    model.load_state_dict(state)
    model.eval()
    return model


@torch.no_grad()
def predict_mask(model, images, device):
    logits = get_logits(model(images.to(device)))
    return logits.argmax(dim=1).cpu()          # [B, H, W], giá trị 0/1


def iou(pred, target):
    inter = torch.logical_and(pred, target).sum().item()
    union = torch.logical_or(pred, target).sum().item()
    return inter / union if union > 0 else float("nan")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-samples", type=int, default=10)
    parser.add_argument("--split", choices=["test", "val"], default="test")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _, val_loader, test_loader = make_dataloaders(download=True)
    loader = test_loader if args.split == "test" else val_loader

    models = {name: load_trained_model(name, device) for name in MODEL_NAMES}

    # Lấy trực tiếp từ dataset để không phụ thuộc batch size.
    dataset = loader.dataset
    n = min(args.num_samples, len(dataset))
    samples = [dataset[i] for i in range(n)]
    images = torch.stack([s[0] for s in samples])
    masks = torch.stack([s[1] for s in samples])

    preds = {name: predict_mask(model, images, device) for name, model in models.items()}

    out_dir = Path(OUTPUT_DIR, "results", "examples")
    out_dir.mkdir(parents=True, exist_ok=True)

    # ---- Tuỳ chọn: lưu mask thô dạng PNG cho từng model ----
    for name in MODEL_NAMES:
        for i in range(n):
            arr = (preds[name][i].numpy() * 255).astype(np.uint8)
            Image.fromarray(arr).save(out_dir / f"{name}_sample{i}.png")

    # ---- Vẽ lưới so sánh: mỗi hàng 1 ảnh, mỗi cột 1 nguồn ----
    cols = ["input", "ground truth"] + list(MODEL_NAMES)
    fig, axes = plt.subplots(n, len(cols), figsize=(2.6 * len(cols), 2.6 * n))
    if n == 1:
        axes = axes[None, :]

    for i in range(n):
        rgb = denormalize(images[i]).permute(1, 2, 0).numpy()

        axes[i, 0].imshow(rgb)
        axes[i, 1].imshow(masks[i].numpy(), cmap="gray", vmin=0, vmax=1)
        if i == 0:
            axes[i, 0].set_title("input")
            axes[i, 1].set_title("ground truth")

        for j, name in enumerate(MODEL_NAMES, start=2):
            p = preds[name][i].numpy()
            axes[i, j].imshow(p, cmap="gray", vmin=0, vmax=1)
            # Muốn overlay viền mask lên ảnh gốc thì thay 2 dòng trên bằng:
            # axes[i, j].imshow(rgb)
            # axes[i, j].contour(p, levels=[0.5], colors="red", linewidths=0.8)
            score = iou(preds[name][i].bool(), masks[i].bool())
            title = f"{name} (IoU={score:.3f})" if i == 0 else f"IoU={score:.3f}"
            axes[i, j].set_title(title)

        for ax in axes[i]:
            ax.axis("off")

    fig.tight_layout()
    out_file = out_dir / f"compare_{args.split}.png"
    fig.savefig(out_file, dpi=150)
    print("Saved:", out_file)


if __name__ == "__main__":
    main()