# demo.py
# Tạo 1 hình so sánh: Original | Ground Truth | Simple | U-Net | Transfer

from pathlib import Path
import numpy as np
import torch
import matplotlib.pyplot as plt
from torchvision.transforms import functional as TF

from config import OUTPUT_DIR
from dataset import make_dataloaders, MEAN, STD
from models import build_model, get_logits
from utils import ensure_dirs


def denormalize(image_tensor):
    mean = torch.tensor(MEAN).view(3, 1, 1)
    std = torch.tensor(STD).view(3, 1, 1)
    image = image_tensor.cpu() * std + mean
    return image.clamp(0, 1)


def main():
    ensure_dirs()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_names = ("simple", "unet", "transfer")
    missing = [
        name for name in model_names
        if not Path(OUTPUT_DIR, "checkpoints", f"{name}_best.pt").exists()
    ]
    if missing:
        raise FileNotFoundError(
            "Comparison figure requires all three checkpoints. Missing: "
            + ", ".join(missing)
        )

    _, _, test_loader = make_dataloaders(download=True)

    images, masks = next(iter(test_loader))
    image = images[0:1].to(device)
    gt = masks[0].cpu().numpy()

    predictions = {}
    for name in model_names:
        checkpoint = Path(OUTPUT_DIR, "checkpoints", f"{name}_best.pt")
        model = build_model(name, pretrained_transfer=False).to(device)
        model.load_state_dict(
            torch.load(checkpoint, map_location=device, weights_only=True)
        )
        model.eval()

        with torch.no_grad():
            logits = get_logits(model(image))
            pred = logits.argmax(dim=1)[0].cpu().numpy()
        predictions[name] = pred

    original = denormalize(images[0]).permute(1, 2, 0).numpy()

    titles = ["Original", "Ground Truth"] + list(predictions.keys())
    arrays = [original, gt] + list(predictions.values())

    fig, axes = plt.subplots(1, len(arrays), figsize=(4 * len(arrays), 4))
    if len(arrays) == 1:
        axes = [axes]

    for ax, title, arr in zip(axes, titles, arrays):
        if title == "Original":
            ax.imshow(arr)
        else:
            ax.imshow(arr, cmap="gray", vmin=0, vmax=1)
        ax.set_title(title)
        ax.axis("off")

    plt.tight_layout()
    out_path = Path(OUTPUT_DIR, "results", "demo_comparison.png")
    plt.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print("Saved:", out_path)


if __name__ == "__main__":
    main()
