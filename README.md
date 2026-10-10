# Pet Segmentation — 3 Model Comparison

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
project-root/
├── src/                     # toàn bộ code Python chính
│   ├── config.py			 # hyperparameters + đường dẫn
│   ├── dataset.py			 # Dataset, splits, và cả preprocessing/augment
│   ├── models.py			 # 3 kiến trúc model
│   ├── utils.py			 # seed, metrics helper
│   ├── train.py			 # script chạy training
│   ├── evaluate.py			 # script chạy evaluating
│   ├── demo.py
│   └── inspect_data.py
├── tests/
│   └── test_core.py		 # test công thức metrics, test model shape, ...
├── docs/                    # ghi chú markdown phụ
│   ├── EXPLAIN_MODELS.md
│   ├── REPORT_OUTLINE.md
│   └── SOURCES.md
├── report/                  # báo cáo + đề bài (docx/pdf)
│   ├── Major_Assignment_Report_Project_2026.docx
│   └── Midterm Requirements _Project.docx
├── outputs/                 # checkpoints, splits.npz, metrics.csv
├── data/                    # dataset 
├── .gitattributes
├── .gitignore
├── LICENSE
├── requirements.txt
└── README.md
```



---

## Kết quả thực nghiệm

Đánh giá trên tập test (100 ảnh), mỗi model dùng checkpoint có IoU validation cao nhất. Toàn bộ metric được tính từ **một confusion matrix mức pixel trên toàn bộ test set**, coi class pet là positive. Với bài toán nhị phân, Dice và F1-score trùng nhau về mặt công thức nên gộp chung một cột.

| Model                                         |              IoU |        Dice / F1 |         Accuracy |        Precision |           Recall |     Params | Inference (ms/ảnh) |
| --------------------------------------------- | ---------------: | ---------------: | ---------------: | ---------------: | ---------------: | ---------: | ------------------: |
| Simple NN (`SimpleSegNet`)                  |           0.6106 |           0.7582 |           0.8043 |           0.7861 |           0.7322 |    106,770 |               11.07 |
| Complex NN (`MiniUNet`)                     |           0.6846 |           0.8128 |           0.8411 |           0.8027 |           0.8231 |    118,930 |               17.80 |
| Transfer Learning (`DeepLabV3-MobileNetV3`) | **0.8240** | **0.9035** | **0.9185** | **0.8967** | **0.9104** | 11,020,594 |               17.89 |

### Nhận xét nhanh

- **Simple NN** cho kết quả thấp nhất (IoU 0.61). Kiến trúc encoder-decoder không có skip connection khiến chi tiết không gian mất dần sau mỗi lần MaxPool; Recall (0.73) thấp hơn Precision (0.79) cho thấy model hay bỏ sót vùng pet, đặc biệt ở phần rìa. Bù lại đây là model nhẹ và nhanh nhất (11.07 ms/ảnh).
- **Complex NN** cải thiện khoảng 7 điểm IoU so với baseline dù cùng độ sâu, nhờ skip connection giúp decoder lấy lại đặc trưng phân giải cao từ encoder. Precision và Recall cân bằng (~0.81 / ~0.82), tức model không thiên về phía bỏ sót hay nhầm nền.
- **Transfer Learning** vượt trội với IoU 0.824, hơn model thứ hai ~14 điểm. Backbone MobileNetV3 pretrain trên ImageNet cung cấp đặc trưng thị giác mạnh mà 600 ảnh train không thể tự học, còn ASPP bổ sung ngữ cảnh đa tỉ lệ. Chi phí là ~11 triệu tham số (gấp ~100 lần hai model còn lại), nhưng thời gian inference thực tế vẫn tương đương (~18 ms/ảnh).

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
python src\inspect_data.py
```

Lần đầu Torchvision sẽ tải Oxford-IIIT Pet.

### 2. Kiểm tra shape 3 model

```bash
python src\models.py
```

Bạn cần thấy output có dạng:

```text
[B, 2, 128, 128]
```

### 3. Train Simple NN

```bash
python src\train.py --model simple
```

### 4. Train Mini U-Net

```bash
python src\train.py --model unet
```

### 5. Train Transfer Learning

```bash
python src\train.py --model transfer
```

Transfer model thực hiện:

- Epoch 1-2: freeze MobileNetV3 backbone, train classifier.
- Epoch 3-10: unfreeze backbone, fine-tune toàn bộ model với learning rate nhỏ hơn.

### 6. Đánh giá cả 3

```bash
python src\evaluate.py
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
python src\demo.py
```

Kết quả:

```text
outputs/results/demo_comparison.png
```

---

## Hyperparameters

| Cấu hình         |              Giá trị |
| ------------------ | ---------------------: |
| Image size         |              128 x 128 |
| Batch size         |                      8 |
| Train / Val / Test |        600 / 100 / 100 |
| Loss               |       CrossEntropyLoss |
| Optimizer          |                   Adam |
| Simple epochs      |                     15 |
| U-Net epochs       |                     15 |
| Transfer epochs    |                     10 |
| Scratch LR         |                   1e-3 |
| Fine-tune LR       |                   1e-4 |
| Augmentation       | Random horizontal flip |

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
