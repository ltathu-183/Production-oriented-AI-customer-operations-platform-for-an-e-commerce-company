# utils.py

import random
from pathlib import Path
import numpy as np
import torch

from config import OUTPUT_DIR


def set_seed(seed=42):
    """Seed Python, NumPy and PyTorch for reproducible experiments."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def seed_worker(worker_id):
    """Seed NumPy and Python inside each DataLoader worker."""
    del worker_id  # The worker-specific value is already in torch.initial_seed().
    worker_seed = torch.initial_seed() % (2 ** 32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def confusion_counts(logits, target):
    """Đếm TP/FP/FN/TN cho class pet (=1)."""
    pred = logits.argmax(dim=1)

    tp = ((pred == 1) & (target == 1)).sum().item()
    fp = ((pred == 1) & (target == 0)).sum().item()
    fn = ((pred == 0) & (target == 1)).sum().item()
    tn = ((pred == 0) & (target == 0)).sum().item()
    return tp, fp, fn, tn


def metrics_from_counts(tp, fp, fn, tn):
    """Compute global binary pixel metrics for the pet/foreground class."""

    def safe_divide(numerator, denominator):
        return numerator / denominator if denominator else 0.0

    precision = safe_divide(tp, tp + fp)
    recall = safe_divide(tp, tp + fn)
    f1 = safe_divide(2 * precision * recall, precision + recall)
    iou = safe_divide(tp, tp + fp + fn)
    dice = safe_divide(2 * tp, 2 * tp + fp + fn)
    accuracy = safe_divide(tp + tn, tp + tn + fp + fn)
    return {
        "iou": iou,
        "dice": dice,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def ensure_dirs():
    Path(OUTPUT_DIR, "checkpoints").mkdir(parents=True, exist_ok=True)
    Path(OUTPUT_DIR, "results").mkdir(parents=True, exist_ok=True)
