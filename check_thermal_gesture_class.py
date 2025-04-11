import os
import torch
from torch.utils.data import Dataset
import pandas as pd
from PIL import Image
import numpy as np
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF


class DualViewCrop(object):
    def __init__(self, crop_size=(194, 194), offset=(0, 0), output_channel_last=True):
        self.crop_size = crop_size
        self.offset = offset
        self.to_gray = transforms.Grayscale(num_output_channels=1)
        self.to_tensor = transforms.ToTensor()


    def __call__(self, img):
        # 假設原始圖像是左右排列的雙視野圖像
        w, h = img.size
        mid = w // 2

        # 左視野裁剪
        left_crop = TF.center_crop(img.crop((0, 0, mid, h)), self.crop_size)

        # 右視野裁剪（含偏移）
        right_start = mid + self.offset[0]
        right_crop = TF.center_crop(img.crop((right_start, self.offset[1], w, h)), self.crop_size)

        # 灰階轉換
        left_crop_gray = self.to_gray(left_crop)
        right_crop_gray = self.to_gray(right_crop)

        # 轉 tensor 並去掉 channel 維度 → [H, W]
        left_tensor = self.to_tensor(left_crop_gray).squeeze(0)
        right_tensor = self.to_tensor(right_crop_gray).squeeze(0)

        stacked = torch.stack([left_tensor, right_tensor], dim=0)   # dim [2, H, W]

        return stacked

# 定義圖像轉換
# img_transform = DualViewCrop(crop_size=(194, 194), offset=(0, 0))

class ThermalGesture(Dataset):
    def __init__(self, img_folder, label_csv, img_transform=None):
        self.folder = img_folder
        self.labels_df = pd.read_csv(label_csv)
        self.img_transform = img_transform
        
        # 檢查文件夾是否存在
        if not os.path.exists(img_folder):
            raise FileNotFoundError(f"圖像文件夾 {img_folder} 不存在")
            
        # 檢查 CSV 文件中的圖像是否都存在
        self._validate_images()

    def _validate_images(self):
        """驗證所有標籤對應的圖像文件是否存在"""
        missing_files = []
        for idx in range(len(self.labels_df)):
            img_number = self.labels_df.iloc[idx, 0]
            img_name = os.path.join(self.folder, f"{img_number:04d}.bmp")
            if not os.path.exists(img_name):
                missing_files.append(img_name)
        
        if missing_files:
            print(f"警告: 找不到 {len(missing_files)} 個圖像文件")
            if len(missing_files) < 10:  # 只顯示前幾個缺失文件
                print(f"缺失文件: {missing_files}")

    def __len__(self):
        return len(self.labels_df)

    def __getitem__(self, idx):
        # 將整數轉換為文件名格式
        img_number = self.labels_df.iloc[idx, 0]
        img_name = os.path.join(self.folder, f"{img_number:04d}.bmp")
        
        try:
            image = Image.open(img_name)
        except Exception as e:
            print(f"無法打開圖像 {img_name}: {e}")
            raise
        
        # 獲取 one-hot 編碼標籤並轉換為張量
        label = torch.tensor(self.labels_df.iloc[idx, 1:].values, dtype=torch.float32)

        # 應用圖像轉換
        image = self.img_transform(image)

        # 修改檢查邏輯
        if image.shape[0] != 2:  # 預期通道數為 2
            print(f"警告: 第 {idx} 張圖像的通道數為 {image.shape[0]}，預期為 2")

        return image, label

# Test
# 定義圖像轉換
img_transform = transforms.Compose([
    DualViewCrop(crop_size=(194, 194), offset=(0, 0)),  # 切分左右視野
    # transforms.ToTensor(),
    # transforms.Normalize(mean=[0.5], std=[0.5])  # GPT建議
])

# 創建數據集
dataset = ThermalGesture(
    img_folder=r'ThermalHand\ThermalGesture',
    label_csv=r'ThermalHand\label.csv',
    img_transform=img_transform
)



import torch.nn as nn

class DualViewCNN(nn.Module):
    def __init__(self, num_classes):
        super(DualViewCNN, self).__init__()
        # 雙視野特徵提取分支（共享權重）
        self.branch = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),  # 194x194 → 97x97
            
            nn.Conv2d(32, 64, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),  # 97x97 → 48x48
            
            nn.Conv2d(64, 128, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2)   # 48x48 → 24x24
        )
        
        # 特徵合併與分類器
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(2 * 128 * 24 * 24, 128),  # 雙視野特徵合併
            nn.ReLU(inplace=True),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        # 分割左右視野 (batch_size, 2, 194, 194)
        left = x[:, 0:1, :, :]  # 左視野 (batch, 1, 194, 194)
        right = x[:, 1:2, :, :] # 右視野 (batch, 1, 194, 194)
        
        # 特徵提取
        left_feat = self.branch(left)  # (batch, 128*24*24)
        right_feat = self.branch(right)
        
        # 特徵合併
        combined = torch.cat([left_feat, right_feat], dim=1)
        return self.classifier(combined)

from torch.utils.data import DataLoader

# 定義模型
num_classes = dataset.labels_df.shape[1] - 1  # 假設標籤的列數減去圖像編號列
model = DualViewCNN(num_classes)

dataloader = DataLoader(dataset, batch_size=8, shuffle=True)
# Test
# for images, labels in dataloader:
#     print(images.shape)  # 應輸出 [batch_size, 2, 194, 194]
#     output = model(images)  # 傳入模型
#     print(output.shape)  # 應輸出 [batch_size, num_classes]

import torch
from torch.utils.data import DataLoader, random_split
import torch.nn as nn
import torch.optim as optim

# 設定隨機種子以便重現結果
import torch
import numpy as np
import random

# 設定隨機種子
def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(42)  # 可以改成你想用的數字

# worker_init_fn：讓每個 DataLoader worker 也使用相同 seed
def worker_init_fn(worker_id):
    worker_seed = 42 + worker_id
    np.random.seed(worker_seed)
    random.seed(worker_seed)

# 設定參數
batch_size = 16
learning_rate = 0.0001
num_epochs = 100
validation_split = 0.2

# 隨機拆分數據集為訓練集和驗證集
dataset_size = len(dataset)
val_size = int(dataset_size * validation_split)
train_size = dataset_size - val_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size],
                                          generator=torch.Generator().manual_seed(42))  #

# 定義加載器並加 seed 控制
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, worker_init_fn=worker_init_fn)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, worker_init_fn=worker_init_fn)

# 定義模型
num_classes = dataset.labels_df.shape[1] - 1  # 假設標籤的列數減去圖像編號列
model = DualViewCNN(num_classes)

# 定義損失函數和優化器
criterion = nn.CrossEntropyLoss()  # 假設標籤是單一類別分類
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

import matplotlib.pyplot as plt

# 訓練模型
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# 儲存每個 epoch 的 Loss 和 Accuracy
train_losses = []
train_accuracies = []
val_losses = []
val_accuracies = []
best_val_loss = float('inf')
for epoch in range(num_epochs):
    # 訓練階段
    model.train()
    train_loss = 0.0
    correct = 0
    total = 0

    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        # 前向傳播
        outputs = model(images)
        loss = criterion(outputs, labels.argmax(dim=1))  # 假設標籤是 one-hot 編碼

        # 反向傳播與優化
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels.argmax(dim=1)).sum().item()

    train_accuracy = 100. * correct / total
    train_losses.append(train_loss / len(train_loader))
    train_accuracies.append(train_accuracy)
    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {train_loss/len(train_loader):.4f}, Accuracy: {train_accuracy:.2f}%")

    # 驗證階段
    model.eval()
    val_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels.argmax(dim=1))

            val_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels.argmax(dim=1)).sum().item()

    val_accuracy = 100. * correct / total
    val_losses.append(val_loss / len(val_loader))
    val_accuracies.append(val_accuracy)
    print(f"Validation Loss: {val_loss/len(val_loader):.4f}, Validation Accuracy: {val_accuracy:.2f}%")

# 儲存最後一個 Epoch 的模型
torch.save(model.state_dict(), "last_epoch_model_epoch_100_batchsize_16.pth")
print("模型已儲存為 'last_epoch_model.pth'")

# 繪製 Loss 和 Accuracy 圖表
epochs = range(1, num_epochs + 1)

plt.figure(figsize=(12, 5))

# Loss 圖表
plt.subplot(1, 2, 1)
plt.plot(epochs, train_losses, label="Train Loss")
plt.plot(epochs, val_losses, label="Validation Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Loss vs. Epoch")
plt.legend()

# Accuracy 圖表
plt.subplot(1, 2, 2)
plt.plot(epochs, train_accuracies, label="Train Accuracy")
plt.plot(epochs, val_accuracies, label="Validation Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy (%)")
plt.title("Accuracy vs. Epoch")
plt.legend()

plt.tight_layout()
plt.show()