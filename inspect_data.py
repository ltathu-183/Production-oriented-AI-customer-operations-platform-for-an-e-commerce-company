# inspect_data.py
# Chạy file này đầu tiên để hiểu dữ liệu trước khi train.

import torch
import matplotlib.pyplot as plt
from dataset import make_dataloaders, MEAN, STD


def denormalize(x):
    mean = torch.tensor(MEAN).view(3, 1, 1)
    std = torch.tensor(STD).view(3, 1, 1)
    return (x * std + mean).clamp(0, 1)


def main():
    train_loader, val_loader, test_loader = make_dataloaders(download=True)
    images, masks = next(iter(train_loader))

    print("Train images:", len(train_loader.dataset))
    print("Val images  :", len(val_loader.dataset))
    print("Test images :", len(test_loader.dataset))
    print("Image shape :", images.shape)
    print("Mask shape  :", masks.shape)
    print("Mask values :", torch.unique(masks))

    image = denormalize(images[0]).permute(1, 2, 0)
    mask = masks[0]

    plt.figure(figsize=(7, 3))
    plt.subplot(1, 2, 1)
    plt.imshow(image)
    plt.title("Image")
    plt.axis("off")

    plt.subplot(1, 2, 2)
    plt.imshow(mask, cmap="gray", vmin=0, vmax=1)
    plt.title("Binary mask")
    plt.axis("off")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
