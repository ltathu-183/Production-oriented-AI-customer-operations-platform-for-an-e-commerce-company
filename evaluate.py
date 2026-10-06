# evaluate.py
# Đánh giá 3 model trên CHÍNH XÁC cùng test set.

import csv
import time
from pathlib import Path
import torch

from config import OUTPUT_DIR
from dataset import make_dataloaders
from models import build_model, get_logits, count_parameters
from utils import confusion_counts, metrics_from_counts, ensure_dirs

MODEL_NAMES = ("simple", "unet", "transfer")


@torch.no_grad()
def evaluate_model(model_name, test_loader, device):
    checkpoint = Path(OUTPUT_DIR, "checkpoints", f"{model_name}_best.pt")
    if not checkpoint.exists():
        print(f"Skip {model_name}: checkpoint not found: {checkpoint}")
        return None

    # Không tải pretrained weights nữa: checkpoint chứa weights sau training.
    model = build_model(model_name, pretrained_transfer=False).to(device)
    state = torch.load(checkpoint, map_location=device, weights_only=True)
    model.load_state_dict(state)
    model.eval()

    total_params, _ = count_parameters(model)
    tp = fp = fn = tn = 0

    # Warm-up vài batch cho timing GPU ổn định hơn.
    if device.type == "cuda":
        images, _ = next(iter(test_loader))
        images = images.to(device)
        for _ in range(3):
            _ = get_logits(model(images))
        torch.cuda.synchronize()

    inference_seconds = 0.0
    num_images = 0

    for images, masks in test_loader:
        images = images.to(device)
        if device.type == "cuda":
            torch.cuda.synchronize()
        start = time.perf_counter()
        logits = get_logits(model(images))
        if device.type == "cuda":
            torch.cuda.synchronize()
        inference_seconds += time.perf_counter() - start

        masks = masks.to(device)
        a, b, c, d = confusion_counts(logits, masks)
        tp += a; fp += b; fn += c; tn += d
        num_images += images.size(0)

    metrics = metrics_from_counts(tp, fp, fn, tn)
    metrics["model"] = model_name
    metrics["params"] = total_params
    metrics["ms_per_image"] = inference_seconds * 1000 / num_images
    return metrics


def main():
    ensure_dirs()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    missing = [
        name for name in MODEL_NAMES
        if not Path(OUTPUT_DIR, "checkpoints", f"{name}_best.pt").exists()
    ]
    if missing:
        commands = "\n".join(f"  python train.py --model {name}" for name in missing)
        raise FileNotFoundError(
            "Evaluation requires checkpoints for all three assignment models. "
            f"Missing: {', '.join(missing)}. Train them first:\n{commands}"
        )

    _, _, test_loader = make_dataloaders(download=True)

    rows = []
    for name in MODEL_NAMES:
        result = evaluate_model(name, test_loader, device)
        rows.append(result)
        print(
            f"{name:8s} | IoU={result['iou']:.4f} | "
            f"Dice={result['dice']:.4f} | "
            f"Acc={result['accuracy']:.4f} | "
            f"Precision={result['precision']:.4f} | "
            f"Recall={result['recall']:.4f} | "
            f"F1={result['f1']:.4f} | "
            f"{result['ms_per_image']:.2f} ms/image | "
            f"params={result['params']:,}"
        )

    out_file = Path(OUTPUT_DIR, "results", "metrics.csv")
    with open(out_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "model", "iou", "dice", "accuracy", "precision", "recall",
                "f1", "params", "ms_per_image",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)
    print("Saved:", out_file)


if __name__ == "__main__":
    main()
