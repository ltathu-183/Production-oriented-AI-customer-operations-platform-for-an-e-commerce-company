# Giải thích 3 model theo cách dễ bảo vệ

## 1. Dữ liệu đi vào model

Một batch có shape:

```text
[B, 3, 128, 128]
```

- `B`: số ảnh trong batch.
- `3`: RGB.
- `128 x 128`: kích thước ảnh.

Ground truth mask:

```text
[B, 128, 128]
```

Mỗi pixel là `0` hoặc `1`.

Mỗi model xuất:

```text
[B, 2, 128, 128]
```

Hai channel cuối là score cho:

- channel 0: background
- channel 1: pet

Dùng `argmax(dim=1)` để lấy class có score lớn nhất cho từng pixel.

---

# 2. Simple NN

Sơ đồ:

```text
Input
  |
Conv 3->16
  |
Pool
  |
Conv 16->32
  |
Pool
  |
Conv 32->64       <- bottleneck
  |
Upsample
  |
Conv 64->32
  |
Upsample
  |
Conv 32->16
  |
1x1 Conv -> 2 classes
```

Encoder làm ảnh feature nhỏ dần để học thông tin tổng quát.
Decoder phóng feature map lớn lại để dự đoán class từng pixel.

Điểm yếu: khi pooling, thông tin vị trí chi tiết có thể bị mất.

---

# 3. Mini U-Net

Mini U-Net gần giống Simple NN nhưng có **skip connection**:

```text
Encoder feature ------------------> Decoder
```

Ví dụ feature `enc1` có thông tin chi tiết ở resolution cao. Khi decoder phóng ảnh lên lại, ta nối (`torch.cat`) feature encoder vào decoder.

Vì vậy Mini U-Net có thể dùng đồng thời:

- feature sâu: hiểu đối tượng là gì;
- feature nông: giữ cạnh và vị trí chi tiết.

Đây là điểm khác biệt chính bạn cần hiểu, không cần học U-Net rất sâu.

---

# 4. Transfer Learning: DeepLabV3 + MobileNetV3

Không tự viết toàn bộ DeepLabV3. Ta dùng implementation có sẵn trong Torchvision.

Ý tưởng:

```text
Image
  |
MobileNetV3 backbone
(pretrained visual features)
  |
DeepLabV3 classifier
  |
2-class segmentation mask
```

## Freeze

```python
for param in model.backbone.parameters():
    param.requires_grad = False
```

Backbone không cập nhật trọng số. Chỉ classifier mới học.

Lý do: classifier mới vừa được thay thành 2 class và cần học trước.

## Fine-tune

Sau vài epoch:

```python
for param in model.backbone.parameters():
    param.requires_grad = True
```

Sau đó dùng learning rate nhỏ hơn để điều chỉnh toàn bộ network cho Oxford Pet.

---

# 5. Loss

Project dùng:

```python
nn.CrossEntropyLoss()
```

Giả sử một pixel đúng là pet (`1`). Model trả hai score:

```text
background = 0.3
pet        = 1.7
```

Cross entropy khuyến khích score của class đúng lớn hơn và class sai nhỏ hơn.

Không cần tự gọi softmax trước `CrossEntropyLoss`.

---

# 6. Backpropagation

Ba dòng quan trọng:

```python
optimizer.zero_grad()
loss.backward()
optimizer.step()
```

- `zero_grad()`: xóa gradient batch trước.
- `backward()`: tính gradient của loss đối với weights.
- `step()`: optimizer dùng gradient để cập nhật weights.

Đây là lõi của training loop.

---

# 7. IoU

Với foreground pet:

```text
IoU = TP / (TP + FP + FN)
```

Nó đo phần giao giữa predicted mask và ground truth chia cho phần hợp.

# 8. Dice

```text
Dice = 2TP / (2TP + FP + FN)
```

Dice cũng đo overlap. Giá trị càng gần 1 càng tốt.

---

# 9. Câu hỏi nghiên cứu

Bạn có thể dùng:

> How do model complexity and transfer learning affect pet image segmentation performance under the same dataset split?

Sau khi chạy thí nghiệm, không kết luận trước model nào tốt hơn. Dựa vào `metrics.csv` rồi giải thích kết quả thực tế.
