# dataset.py
# Mục tiêu: tải Oxford-IIIT Pet, tạo mask nhị phân và đảm bảo
# 3 model dùng chính xác cùng train/val/test split.

from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.datasets import OxfordIIITPet
from torchvision.transforms import functional as TF
from torchvision.transforms import InterpolationMode

from config import (
    SEED, IMAGE_SIZE, N_TRAIN, N_VAL, N_TEST,
    BATCH_SIZE, NUM_WORKERS, DATA_DIR, OUTPUT_DIR
)
from utils import seed_worker

# ImageNet mean/std. Dùng cùng preprocessing cho cả 3 model.
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]


class PetSegmentationDataset(Dataset):
    """Wrapper đơn giản quanh torchvision OxfordIIITPet.

    Mask gốc có 3 giá trị:
        1 = pet
        2 = background
        3 = border

    Để project dễ hiểu, ta gộp border vào pet:
        background -> 0
        pet + border -> 1
    """

    def __init__(self, root, split, indices, train=False, download=False):
        self.base = OxfordIIITPet(
            root=root,
            split=split,
            target_types="segmentation",
            download=download,
        )
        self.indices = list(indices)
        self.train = train

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        image, trimap = self.base[self.indices[idx]]

        # Resize ảnh bằng bilinear vì ảnh RGB là dữ liệu liên tục.
        image = TF.resize(
            image,
            [IMAGE_SIZE, IMAGE_SIZE],
            interpolation=InterpolationMode.BILINEAR,
        )

        # Resize mask bằng nearest để không sinh giá trị nhãn mới.
        trimap = TF.resize(
            trimap,
            [IMAGE_SIZE, IMAGE_SIZE],
            interpolation=InterpolationMode.NEAREST,
        )

        # Augmentation duy nhất: flip ngang, áp dụng đồng thời ảnh + mask.
        if self.train and torch.rand(1).item() < 0.5:
            image = TF.hflip(image)
            trimap = TF.hflip(trimap)

        # PIL image -> tensor [3, H, W], giá trị [0, 1].
        image = TF.to_tensor(image)
        image = TF.normalize(image, mean=MEAN, std=STD)

        # trimap: 1=pet, 2=background, 3=border.
        raw = np.array(trimap, dtype=np.uint8)
        mask = (raw != 2).astype(np.int64)  # pet + border = 1
        mask = torch.from_numpy(mask)       # [H, W], dtype long

        return image, mask


def create_or_load_splits(download=True):
    """Tạo split cố định và lưu ra file để mọi model dùng cùng dữ liệu."""
    split_dir = Path(OUTPUT_DIR)
    split_dir.mkdir(parents=True, exist_ok=True)
    split_file = split_dir / "splits.npz"

    # Gọi dataset gốc để biết số lượng ảnh.
    trainval_base = OxfordIIITPet(
        root=DATA_DIR,
        split="trainval",
        target_types="segmentation",
        download=download,
    )
    test_base = OxfordIIITPet(
        root=DATA_DIR,
        split="test",
        target_types="segmentation",
        download=download,
    )

    if split_file.exists():
        with np.load(split_file) as data:
            required = {"train", "val", "test"}
            if set(data.files) != required:
                raise ValueError(
                    f"Invalid split file {split_file}: expected arrays {sorted(required)}. "
                    "Delete it to regenerate the splits."
                )
            train_indices = data["train"].copy()
            val_indices = data["val"].copy()
            test_indices = data["test"].copy()

        if len(train_indices) != N_TRAIN or len(val_indices) != N_VAL or len(test_indices) != N_TEST:
            raise ValueError(
                f"Saved splits in {split_file} do not match the current configuration "
                f"({N_TRAIN}/{N_VAL}/{N_TEST}). Delete the file to regenerate them."
            )
        if not all(
            np.issubdtype(indices.dtype, np.integer)
            for indices in (train_indices, val_indices, test_indices)
        ):
            raise ValueError(f"Saved split file {split_file} must contain integer indices.")
        if (
            np.any(train_indices < 0) or np.any(train_indices >= len(trainval_base))
            or np.any(val_indices < 0) or np.any(val_indices >= len(trainval_base))
            or np.any(test_indices < 0) or np.any(test_indices >= len(test_base))
            or np.intersect1d(train_indices, val_indices).size
            or np.unique(train_indices).size != len(train_indices)
            or np.unique(val_indices).size != len(val_indices)
            or np.unique(test_indices).size != len(test_indices)
        ):
            raise ValueError(f"Saved split file {split_file} contains invalid or overlapping indices.")
        return train_indices, val_indices, test_indices

    rng = np.random.default_rng(SEED)

    # Train và validation lấy từ split trainval chính thức.
    trainval_indices = rng.permutation(len(trainval_base))
    if N_TRAIN + N_VAL > len(trainval_indices):
        raise ValueError("N_TRAIN + N_VAL exceeds the available trainval images.")

    if N_TEST > len(test_base):
        raise ValueError("N_TEST is larger than the number of available test images.")

    train_indices = trainval_indices[:N_TRAIN]
    val_indices = trainval_indices[N_TRAIN:N_TRAIN + N_VAL]

    # Test lấy từ split test chính thức, hoàn toàn tách biệt.
    test_indices = rng.permutation(len(test_base))[:N_TEST]

    np.savez(
        split_file,
        train=train_indices,
        val=val_indices,
        test=test_indices,
    )
    return train_indices, val_indices, test_indices


def make_dataloaders(download=True):
    train_idx, val_idx, test_idx = create_or_load_splits(download=download)

    train_ds = PetSegmentationDataset(
        DATA_DIR, "trainval", train_idx, train=True, download=False
    )
    val_ds = PetSegmentationDataset(
        DATA_DIR, "trainval", val_idx, train=False, download=False
    )
    test_ds = PetSegmentationDataset(
        DATA_DIR, "test", test_idx, train=False, download=False
    )

    common = dict(
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        pin_memory=torch.cuda.is_available(),
        worker_init_fn=seed_worker,
    )

    # The generator controls both the shuffled order and worker base seeds.
    train_generator = torch.Generator().manual_seed(SEED)
    train_loader = DataLoader(
        train_ds,
        shuffle=True,
        generator=train_generator,
        **common,
    )
    val_loader = DataLoader(val_ds, shuffle=False, **common)
    test_loader = DataLoader(test_ds, shuffle=False, **common)

    return train_loader, val_loader, test_loader


if __name__ == "__main__":
    train_loader, val_loader, test_loader = make_dataloaders(download=True)
    images, masks = next(iter(train_loader))
    print("Train/Val/Test:", len(train_loader.dataset), len(val_loader.dataset), len(test_loader.dataset))
    print("Image batch:", images.shape, images.dtype)
    print("Mask batch :", masks.shape, masks.dtype)
    print("Mask values:", torch.unique(masks))
