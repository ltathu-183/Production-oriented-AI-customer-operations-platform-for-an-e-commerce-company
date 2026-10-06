# Pet Segmentation — 3 Model Comparison

Project giữa kỳ xử lý ảnh / deep learning, thiết kế để hoàn thành trong khoảng 1 tuần và có thể hiểu từng phần code.

## Mục tiêu

Dùng **cùng Oxford-IIIT Pet dataset**, cùng train/validation/test split và cùng metric để so sánh:

1. **Simple NN** — `SimpleSegNet`: encoder-decoder CNN cơ bản, không skip connection.
2. **Complex NN** — `MiniUNet`: encoder-decoder có skip connection.
3. **Transfer Learning / Fine-tuning** — DeepLabV3 + MobileNetV3-Large pretrained từ Torchvision.

Bài toán là **binary semantic segmentation**:

- `0 = background`
- `1 = pet`

Để đơn giản, border trong trimap gốc được gộp vào foreground.

---

## Dataset split

Tất cả model dùng chính xác cùng split:

- Train: 600 ảnh từ split `trainval`
- Validation: 100 ảnh từ split `trainval`
- Test: 100 ảnh từ split `test` chính thức
- Seed: 42

Split được lưu tại `outputs/splits.npz`. Vì vậy train model thứ 2/3 không tạo split mới.

---

## Cấu trúc thư mục

```text
pet_segmentation_3models/
├── config.py          # Hyperparameters
├── dataset.py         # Dataset + split + DataLoader
├── models.py          # 3 model
├── utils.py           # IoU/Dice helpers
├── inspect_data.py    # Xem ảnh/mask trước khi train
├── train.py           # Train từng model
├── evaluate.py        # So sánh 3 model trên cùng test set
├── demo.py            # Vẽ kết quả trực quan
├── EXPLAIN_MODELS.md  # Giải thích kiến trúc bằng lời
├── REPORT_OUTLINE.md  # Khung báo cáo
├── requirements.txt
├── data/
└── outputs/
```

---

## Cài đặt

Khuyến nghị tạo virtual environment. Cài PyTorch theo CUDA phù hợp với máy trước, sau đó:

```bash
pip install -r requirements.txt
```

Kiểm tra GPU:

```bash
python -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
```

---

## Thứ tự chạy nên dùng

### 1. Hiểu dữ liệu

```bash
python inspect_data.py
```

Lần đầu Torchvision sẽ tải Oxford-IIIT Pet.

### 2. Kiểm tra shape 3 model

```bash
python models.py
```

Bạn cần thấy output có dạng:

```text
[B, 2, 128, 128]
```

### 3. Train Simple NN

```bash
python train.py --model simple
```

### 4. Train Mini U-Net

```bash
python train.py --model unet
```

### 5. Train Transfer Learning

```bash
python train.py --model transfer
```

Transfer model thực hiện:

- Epoch 1-2: freeze MobileNetV3 backbone, train classifier.
- Epoch 3-10: unfreeze backbone, fine-tune toàn bộ model với learning rate nhỏ hơn.

### 6. Đánh giá cả 3

```bash
python evaluate.py
```

Kết quả được lưu:

```text
outputs/results/metrics.csv
```

Metric chính:

- IoU
- Dice
- Precision
- Recall
- F1-score

All segmentation metrics are computed from one pixel-level confusion matrix
over the complete test set, with pet as the positive class. For this binary
setup, F1-score and Dice are algebraically equivalent and therefore have the
same value.

File cũng ghi Accuracy, số parameter và thời gian inference để tham khảo.

### 7. Tạo ảnh demo

```bash
python demo.py
```

Kết quả:

```text
outputs/results/demo_comparison.png
```

---

## Hyperparameters

| Cấu hình | Giá trị |
|---|---:|
| Image size | 128 x 128 |
| Batch size | 8 |
| Train / Val / Test | 600 / 100 / 100 |
| Loss | CrossEntropyLoss |
| Optimizer | Adam |
| Simple epochs | 15 |
| U-Net epochs | 15 |
| Transfer epochs | 10 |
| Scratch LR | 1e-3 |
| Fine-tune LR | 1e-4 |
| Augmentation | Random horizontal flip |

Nếu GPU 8 GB bị OOM ở transfer model, chỉ cần đổi `BATCH_SIZE = 4` trong `config.py`. Dataset split không thay đổi.

---

## Fair comparison

Ba model dùng:

- cùng ảnh train/val/test;
- cùng 128x128;
- cùng normalization;
- cùng CrossEntropyLoss;
- cùng IoU/Dice implementation;
- cùng test set.

Số epoch và learning rate không bắt buộc giống hệt nhau vì transfer learning cần fine-tuning khác model train-from-scratch. Hãy ghi rõ điều này trong báo cáo.

---

## Điều cần hiểu để bảo vệ

1. Tensor ảnh `[B, 3, H, W]` nghĩa là gì?
2. Tại sao output là `[B, 2, H, W]`?
3. `CrossEntropyLoss` làm gì?
4. `loss.backward()` làm gì?
5. `optimizer.step()` làm gì?
6. Simple CNN khác Mini U-Net ở skip connection như thế nào?
7. Freeze backbone là gì?
8. Fine-tuning khác train from scratch thế nào?
9. IoU và Dice đo cái gì?
10. Tại sao test set phải giống nhau giữa 3 model?

Đọc `EXPLAIN_MODELS.md` trước khi chỉnh thêm kiến trúc.
