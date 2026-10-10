from pathlib import Path

# config.py
# Tất cả tham số chính của project gom vào 1 file để dễ đọc và chỉnh.

SEED = 42
IMAGE_SIZE = 128

# Dùng cùng một split cho cả 3 model.
N_TRAIN = 600
N_VAL = 100
N_TEST = 100

BATCH_SIZE = 8
NUM_WORKERS = 2
NUM_CLASSES = 2  # 0 = background, 1 = pet

# Train from scratch
SIMPLE_EPOCHS = 15
UNET_EPOCHS = 15
LR_SCRATCH = 1e-3

# Transfer learning / fine-tuning
TRANSFER_EPOCHS = 10
FREEZE_EPOCHS = 2
LR_HEAD = 1e-3
LR_FINETUNE = 1e-4

# Resolve project data independently of the caller's current working directory.
PROJECT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_DIR / "data"
OUTPUT_DIR = PROJECT_DIR / "outputs"
