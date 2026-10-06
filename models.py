# models.py
# Gồm đúng 3 model theo yêu cầu:
# 1) Simple CNN: encoder-decoder cơ bản
# 2) Mini U-Net: thêm skip connection
# 3) DeepLabV3-MobileNetV3: transfer learning / fine-tuning

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.segmentation import (
    deeplabv3_mobilenet_v3_large,
    DeepLabV3_MobileNet_V3_Large_Weights,
)

from config import NUM_CLASSES


class ConvBlock(nn.Module):
    """Hai phép Conv + ReLU. Không BatchNorm để giữ code dễ hiểu."""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class SimpleSegNet(nn.Module):
    """Simple neural network: encoder-decoder không có skip connection."""

    def __init__(self, num_classes=NUM_CLASSES):
        super().__init__()
        self.pool = nn.MaxPool2d(2)

        self.enc1 = ConvBlock(3, 16)    # 128 -> 128
        self.enc2 = ConvBlock(16, 32)   # 64 -> 64
        self.bottleneck = ConvBlock(32, 64)  # 32 -> 32

        self.dec2 = ConvBlock(64, 32)
        self.dec1 = ConvBlock(32, 16)
        self.out = nn.Conv2d(16, num_classes, kernel_size=1)

    def forward(self, x):
        x = self.enc1(x)
        x = self.pool(x)
        x = self.enc2(x)
        x = self.pool(x)

        x = self.bottleneck(x)

        x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)
        x = self.dec2(x)
        x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)
        x = self.dec1(x)

        return self.out(x)


class MiniUNet(nn.Module):
    """Complex NN vừa đủ: cùng độ sâu với SimpleSegNet nhưng có skip connection."""

    def __init__(self, num_classes=NUM_CLASSES):
        super().__init__()
        self.pool = nn.MaxPool2d(2)

        self.enc1 = ConvBlock(3, 16)
        self.enc2 = ConvBlock(16, 32)
        self.bottleneck = ConvBlock(32, 64)

        # Sau concat: 64 + 32 = 96 channels.
        self.dec2 = ConvBlock(64 + 32, 32)
        # Sau concat: 32 + 16 = 48 channels.
        self.dec1 = ConvBlock(32 + 16, 16)

        self.out = nn.Conv2d(16, num_classes, kernel_size=1)

    def forward(self, x):
        # Encoder: giữ lại feature map cho skip connection.
        s1 = self.enc1(x)             # [B,16,128,128]
        s2 = self.enc2(self.pool(s1)) # [B,32,64,64]
        x = self.bottleneck(self.pool(s2))  # [B,64,32,32]

        # Decoder + skip connection.
        x = F.interpolate(x, size=s2.shape[-2:], mode="bilinear", align_corners=False)
        x = torch.cat([x, s2], dim=1)
        x = self.dec2(x)

        x = F.interpolate(x, size=s1.shape[-2:], mode="bilinear", align_corners=False)
        x = torch.cat([x, s1], dim=1)
        x = self.dec1(x)

        return self.out(x)


def build_transfer_model(pretrained=True, num_classes=NUM_CLASSES):
    """DeepLabV3 + MobileNetV3-Large.

    Khi pretrained=True:
      - tải weights segmentation pretrained của Torchvision
      - thay lớp cuối thành 2 class của bài toán này

    Khi pretrained=False:
      - chỉ tạo đúng kiến trúc để load checkpoint đã train
      - không cần tải pretrained weights lại
    """
    if pretrained:
        weights = DeepLabV3_MobileNet_V3_Large_Weights.DEFAULT
        model = deeplabv3_mobilenet_v3_large(weights=weights)

        # Không dùng auxiliary head để training loop đơn giản.
        model.aux_classifier = None

        # Lớp cuối của classifier là Conv2d(256, 21, 1) ở weights gốc.
        # Thay thành 2 class: background và pet.
        in_channels = model.classifier[-1].in_channels
        model.classifier[-1] = nn.Conv2d(in_channels, num_classes, kernel_size=1)
    else:
        model = deeplabv3_mobilenet_v3_large(
            weights=None,
            weights_backbone=None,
            num_classes=num_classes,
            aux_loss=False,
        )

    return model


def build_model(name, pretrained_transfer=True):
    name = name.lower()
    if name == "simple":
        return SimpleSegNet()
    if name == "unet":
        return MiniUNet()
    if name == "transfer":
        return build_transfer_model(pretrained=pretrained_transfer)
    raise ValueError("Model name must be one of: simple, unet, transfer")


def get_logits(model_output):
    """Simple/U-Net trả Tensor; DeepLab trả dict {'out': Tensor}."""
    if isinstance(model_output, dict):
        return model_output["out"]
    return model_output


def set_backbone_trainable(model, trainable):
    """Freeze/unfreeze backbone của transfer model."""
    for param in model.backbone.parameters():
        param.requires_grad = trainable


def count_parameters(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


if __name__ == "__main__":
    x = torch.randn(2, 3, 128, 128)
    for name in ["simple", "unet", "transfer"]:
        # Không tải pretrained weights khi chỉ test shape.
        model = build_model(name, pretrained_transfer=False)
        y = get_logits(model(x))
        total, trainable = count_parameters(model)
        print(name, "->", y.shape, "params:", total, "trainable:", trainable)
