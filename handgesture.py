# import torch
# print(torch.cuda.is_available())
"""
torchvision.DatasetLoader
label : csv
img_dir : folder 
from torchvision.transforms import v2 / compose
class Thermalgesture(folder,label_csv,img_transform,label_transform)
    def __get_item__()
"""
import os
import pandas as pd
import torch
import torchvision.transforms.functional as TF
from torchvision import transforms
from PIL import Image
from torch.utils.data import Dataset

class DualViewCrop(object):
    def __init__(self, crop_size, offset):
        self.crop_size = crop_size
        self.offset = offset

    def __call__(self, img):
        # 假設原始圖像是左右排列的雙視野圖像
        w, h = img.size
        mid = w // 2

        # 左視野裁剪
        left_crop = TF.center_crop(img.crop((0, 0, mid, h)), self.crop_size)
        # 轉換為灰度圖
        left_crop_gray = TF.to_grayscale(left_crop)
        
        # 右視野裁剪（帶偏移）
        right_start = mid + self.offset[0]
        right_crop = TF.center_crop(img.crop((right_start, self.offset[1], w, h)), self.crop_size)
        # 轉換為灰度圖
        right_crop_gray = TF.to_grayscale(right_crop)
        
        # 將兩個灰度裁剪區域堆疊在一起
        return torch.cat([TF.to_tensor(left_crop_gray), TF.to_tensor(right_crop_gray)], dim=0)

# 定義圖像轉換 - 注意不再需要單獨的 Grayscale 轉換
img_transform = DualViewCrop(crop_size=(128, 128), offset=(0, 0))


class ThermalGesture(Dataset):
    def __init__(self, folder, label_csv, img_transform=None, label_transform=None):
        self.folder = folder
        self.labels_df = pd.read_csv(label_csv)
        self.img_transform = img_transform
        self.label_transform = label_transform

    def __len__(self):
        return len(self.labels_df)

    def __getitem__(self, idx):
        # 將整數轉換為文件名格式
        img_number = self.labels_df.iloc[idx, 0]
        img_name = os.path.join(self.folder, f"{img_number:04d}.bmp")
        
        image = Image.open(img_name)
        label = self.labels_df.iloc[idx, 1:].values.astype(float)

        if self.img_transform:
            image = self.img_transform(image)
        
        if self.label_transform:
            label = self.label_transform(label)

        return image, label

# 定義標籤轉換
def label_to_tensor(x):
    return torch.tensor(x, dtype=torch.float32)

label_transform = transforms.Lambda(label_to_tensor)

# # 創建數據集實例
# dataset = ThermalGesture(
#     folder='ThermalHand\ThermalGesture',
#     label_csv='ThermalHand\label.csv',
#     img_transform=img_transform,
#     label_transform=label_transform
# )

# # 使用 DataLoader
# dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torch.nn.functional as F
import numpy as np
from torch.utils.data import random_split
# 定義 PyTorch 模型
class GestureClassifier(nn.Module):
    def __init__(self, num_classes, input_channels=2):  # 2個視野通道
        super(GestureClassifier, self).__init__()

        # 卷積層塊 1
        self.conv1 = nn.Conv2d(input_channels, 32, kernel_size=5, padding=2)
        self.pool1 = nn.MaxPool2d(kernel_size=2, padding=0)
        
        # 卷積層塊 2
        self.conv2 = nn.Conv2d(32, 64, kernel_size=5, padding=2)
        self.pool2 = nn.MaxPool2d(kernel_size=2, padding=0)
        
        # 卷積層塊 3
        self.conv3 = nn.Conv2d(64, 128, kernel_size=5, padding=2)
        self.pool3 = nn.MaxPool2d(kernel_size=2, padding=0)

        # 加入 dropout 防止 overfitting
        self.dropout = nn.Dropout(p=0.5)
        
        # 假設輸入為 (B, 2, 128, 128) -> (B, 128, 16, 16)
        self.fc1 = nn.Linear(128 * 16 * 16, 128)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool1(x)
        
        x = F.relu(self.conv2(x))
        x = self.pool2(x)
        
        x = F.relu(self.conv3(x))
        x = self.pool3(x)

        x = x.view(x.size(0), -1)
        x = self.dropout(x)  # Dropout 加在展平後
        x = F.relu(self.fc1(x))
        x = self.fc2(x)

        return x  

# 訓練函數
def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=10, device='cuda'):
    model = model.to(device)
    best_val_acc = 0.0
    
    for epoch in range(num_epochs):
        # 訓練階段
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            # 梯度歸零
            optimizer.zero_grad()
            
            # 前向傳播
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            # 反向傳播和優化
            loss.backward()
            optimizer.step()
            
            # 統計
            running_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs, 1)
            _, labels_max = torch.max(labels, 1)  # 假設標籤是 one-hot 編碼
            total += labels.size(0)
            correct += (predicted == labels_max).sum().item()
        
        epoch_loss = running_loss / len(train_loader.dataset)
        epoch_acc = correct / total
        
        # 驗證階段
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item() * inputs.size(0)
                _, predicted = torch.max(outputs, 1)
                _, labels_max = torch.max(labels, 1)  # 假設標籤是 one-hot 編碼
                val_total += labels.size(0)
                val_correct += (predicted == labels_max).sum().item()
        
        val_loss = val_loss / len(val_loader.dataset)
        val_acc = val_correct / val_total
        
        print(f'Epoch {epoch+1}/{num_epochs}')
        print(f'Train Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')
        print(f'Val Loss: {val_loss:.4f} Acc: {val_acc:.4f}')
        
        # 保存最佳模型
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), 'best_model.pth')
    
    return model

def main():
    # 設置隨機種子以確保可重現性
    torch.manual_seed(42)
    np.random.seed(42)
    
    # 檢查是否有可用的 GPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    
    # 創建完整數據集
    full_dataset = ThermalGesture(
        folder='train_dataset\ThermalGesture',
        label_csv='train_dataset\label.csv',
        img_transform=img_transform,
        label_transform=label_transform
    )
    
    # 獲取類別數量 - 從 label.csv 可以看出有 13 個類別
    num_classes = full_dataset.labels_df.iloc[:, 1:].shape[1]  # 或者使用 full_dataset.labels_df.iloc[:, 1:].shape[1]
    
    # 從完整數據集中分割訓練集和驗證集
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
    
    # 創建數據加載器
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=0)
    
    # 創建模型
    model = GestureClassifier(num_classes=num_classes)
    
    # 定義損失函數和優化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # 訓練模型
    model = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        num_epochs=30,
        device=device
    )
    
    # 保存最終模型
    torch.save(model.state_dict(), 'final_model_epoch_30.pth')
    print('Training completed!')

if __name__ == '__main__':
    main()



