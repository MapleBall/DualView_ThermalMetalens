import torch
import numpy as np
import random
from torch.utils.data import DataLoader, random_split
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from torch.optim.lr_scheduler import ReduceLROnPlateau

from dataset import DualViewCrop, ThermalGesture  # 引用 dataset.py module 

# ✅ 設定隨機種子
def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(42)

def worker_init_fn(worker_id):
    seed = 42 + worker_id
    np.random.seed(seed)
    random.seed(seed)

# ✅ 定義模型架構
class DualViewCNN(nn.Module):
    def __init__(self, num_classes):
        super(DualViewCNN, self).__init__()
        self.branch = nn.Sequential(
            nn.Conv2d(1, 32, 5, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, 5, padding=2),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, 5, padding=2),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.AdaptiveAvgPool2d((4, 4))  # ✅ 自動轉成固定大小輸出
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(2 * 128 * 4 * 4, 512),
            nn.ReLU(),
            nn.Dropout(0.25),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Dropout(0.25),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        left, right = x[:, 0:1], x[:, 1:2]
        left_feat = self.branch(left)
        right_feat = self.branch(right)
        combined = torch.cat([left_feat, right_feat], dim=1)
        return self.classifier(combined)

# ✅ 資料與模型初始化
transform = DualViewCrop(
    crop_size=(128, 128),
    auto_crop=True,  # ✅ 開啟自動手勢裁切
    augment=True,
    augment_prob=0.5,
    seed=42
)


dataset = ThermalGesture("ThermalHand\ThermalGesture", "ThermalHand\label.csv", transform)
num_classes = dataset.labels_df.shape[1] - 1
model = DualViewCNN(num_classes)

batch_size = 16
learning_rate = 0.0001
num_epochs = 50
val_split = 0.2

train_size = int(len(dataset) * (1 - val_split))
val_size = len(dataset) - train_size
train_ds, val_ds = random_split(dataset, [train_size, val_size], generator=torch.Generator().manual_seed(42))

train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, worker_init_fn=worker_init_fn)
val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, worker_init_fn=worker_init_fn)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=2, min_lr=1e-6)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

train_losses, val_losses = [], []
train_accuracies, val_accuracies = [], []
best_val_loss = float('inf')

# ✅ 訓練迴圈
for epoch in range(num_epochs):
    model.train()
    total, correct, running_loss = 0, 0, 0.0
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        loss = criterion(outputs, labels.argmax(dim=1))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels.argmax(dim=1)).sum().item()

    train_acc = 100 * correct / total
    train_losses.append(running_loss / len(train_loader))
    train_accuracies.append(train_acc)

    # 驗證階段
    model.eval()
    val_loss, val_correct, val_total = 0.0, 0, 0
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels.argmax(dim=1))
            val_loss += loss.item()
            _, predicted = outputs.max(1)
            val_total += labels.size(0)
            val_correct += predicted.eq(labels.argmax(dim=1)).sum().item()

    val_acc = 100 * val_correct / val_total
    val_loss_avg = val_loss / len(val_loader)
    val_losses.append(val_loss_avg)
    val_accuracies.append(val_acc)

    print(f"\nEpoch {epoch+1}/{num_epochs}")
    print(f"Train Loss: {train_losses[-1]:.4f}, Accuracy: {train_acc:.2f}%")
    print(f"Val   Loss: {val_loss_avg:.4f}, Accuracy: {val_acc:.2f}%")

    scheduler.step(val_loss_avg)

    if val_loss_avg < best_val_loss:
        best_val_loss = val_loss_avg
        torch.save(model, 'best_model.pth')
        print("✅ Best model saved!")

# 📈 畫圖
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(train_losses, label="Train Loss")
plt.plot(val_losses, label="Val Loss")
plt.legend(); plt.title("Loss")

plt.subplot(1, 2, 2)
plt.plot(train_accuracies, label="Train Acc")
plt.plot(val_accuracies, label="Val Acc")
plt.legend(); plt.title("Accuracy")

plt.tight_layout()
plt.show()
