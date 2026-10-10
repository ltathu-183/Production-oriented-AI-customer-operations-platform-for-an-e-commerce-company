# Khung báo cáo giữa kỳ

## 1. Introduction

- Semantic segmentation là gì?
- Bài toán: tách pet khỏi background.
- Mục tiêu: so sánh Simple NN, Complex NN và Transfer Learning trên cùng dữ liệu.

### Research Question

How do model complexity and transfer learning affect pet image segmentation performance under the same dataset split?

## 2. Dataset

- Oxford-IIIT Pet.
- Input image + segmentation trimap.
- Chuyển về binary segmentation: background=0, pet+border=1.
- Resize 128x128.
- Split: 600 train, 100 validation, 100 official-test samples.
- Seed 42.

## 3. Models

### 3.1 Simple Encoder-Decoder CNN

Mô tả encoder, pooling, bottleneck, upsampling.

### 3.2 Mini U-Net

Mô tả giống Simple CNN nhưng thêm skip connections.

### 3.3 DeepLabV3-MobileNetV3 Transfer Learning

- Pretrained model.
- Thay output head thành 2 classes.
- Freeze backbone 2 epochs.
- Unfreeze và fine-tune.

## 4. Training Setup

- CrossEntropyLoss.
- Adam.
- Image size 128x128.
- Batch size 8.
- Simple/U-Net: lr=1e-3.
- Transfer fine-tuning: lr=1e-4.
- Horizontal flip augmentation.

## 5. Metrics

### IoU

IoU = TP / (TP + FP + FN)

### Dice

Dice = 2TP / (2TP + FP + FN)

### Accuracy, Precision, Recall and F1-score

- Accuracy = (TP + TN) / (TP + TN + FP + FN)
- Precision = TP / (TP + FP)
- Recall = TP / (TP + FN)
- F1 = 2 x Precision x Recall / (Precision + Recall)

Metrics are aggregated over all test pixels, with pet as the positive class.
For this binary setup, F1-score and Dice are algebraically equivalent.

## 6. Results

Điền số thật từ `outputs/results/metrics.csv`.

| Model | Accuracy | Precision | Recall | F1 | IoU | Dice | Params | ms/image |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Simple NN | | | | | | | | |
| Mini U-Net | | | | | | | | |
| Transfer Learning | | | | | | | | |

Chèn `outputs/results/demo_comparison.png`.

## 7. Discussion

Phân tích dựa trên kết quả thật:

- Skip connection ảnh hưởng thế nào?
- Pretrained features có hữu ích với dataset nhỏ không?
- Model nào tốn parameter/thời gian hơn?
- Các trường hợp thất bại: nền phức tạp, pet cùng màu nền, pet ở sát biên...

## 8. Conclusion

Tóm tắt điều quan sát được từ 3 model, không chỉ nói model nào có metric cao nhất.

## 9. Limitations

- Chỉ dùng subset 800 ảnh.
- Resolution thấp 128x128.
- Gộp border vào foreground.
- Hyperparameter tuning tối thiểu do giới hạn thời gian.
