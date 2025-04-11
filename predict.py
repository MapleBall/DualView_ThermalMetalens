import os
import torch
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import torch.nn as nn
import torchvision.transforms.functional as TF
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# 預處理：切左右視野 + 灰階
class DualViewCrop(object):
    def __init__(self, crop_size=(194, 194), offset=(0, 0)):
        self.crop_size = crop_size
        self.offset = offset
        self.to_gray = transforms.Grayscale(num_output_channels=1)
        self.to_tensor = transforms.ToTensor()

    def __call__(self, img):
        w, h = img.size
        mid = w // 2

        left_crop = TF.center_crop(img.crop((0, 0, mid, h)), self.crop_size)
        right_start = mid + self.offset[0]
        right_crop = TF.center_crop(img.crop((right_start, self.offset[1], w, h)), self.crop_size)

        left_crop_gray = self.to_gray(left_crop)
        right_crop_gray = self.to_gray(right_crop)

        left_tensor = self.to_tensor(left_crop_gray).squeeze(0)
        right_tensor = self.to_tensor(right_crop_gray).squeeze(0)

        return torch.stack([left_tensor, right_tensor], dim=0)

# 資料集
class ThermalGesture(Dataset):
    def __init__(self, img_folder, label_csv, img_transform=None):
        self.folder = img_folder
        self.labels_df = pd.read_csv(label_csv, encoding="utf-8-sig")
        self.labels_df.iloc[:, 0] = self.labels_df.iloc[:, 0].astype(int)

        self.img_transform = img_transform

    def __len__(self):
        return len(self.labels_df)

    def __getitem__(self, idx):
        img_number = int(self.labels_df.iloc[idx, 0])
        img_name = os.path.join(self.folder, f"{img_number:04d}.bmp")

        image = Image.open(img_name)

        label = torch.tensor(self.labels_df.iloc[idx, 1:].values, dtype=torch.float32)
        image = self.img_transform(image)
        return image, label

# 模型架構（需跟訓練時一致）
class DualViewCNN(nn.Module):
    def __init__(self, num_classes):
        super(DualViewCNN, self).__init__()
        self.branch = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2)
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(2 * 128 * 24 * 24, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        left = x[:, 0:1, :, :]
        right = x[:, 1:2, :, :]
        left_feat = self.branch(left)
        right_feat = self.branch(right)
        combined = torch.cat([left_feat, right_feat], dim=1)
        return self.classifier(combined)

# label 類別
label_list = ["zero", "one", "two", "three", "four", "five", "six", 
              "seven", "eight", "nine", "like", "ok", "love"]

# 路徑設定
image_folder = r"ThermalHand\ThermalGesture"
label_csv = r"ThermalHand\label.csv"
model_path = "last_epoch_model_epoch_200_batchsize_16.pth"

# 數據集和 DataLoader
transform = DualViewCrop(crop_size=(194, 194), offset=(0, 0))
test_dataset = ThermalGesture(image_folder, label_csv, transform)
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

# 載入模型
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = DualViewCNN(num_classes=13)
model.load_state_dict(torch.load(model_path, map_location=device))
model.to(device)
model.eval()

# 預測與混淆矩陣
y_true = []
y_pred = []

with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
        true_labels = labels.argmax(dim=1)

        y_true.extend(true_labels.cpu().numpy())
        y_pred.extend(predicted.cpu().numpy())

# 計算混淆矩陣
cm = confusion_matrix(y_true, y_pred, labels=list(range(len(label_list))))
cm_normalized = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]

# 繪圖
plt.figure(figsize=(10, 8))
sns.heatmap(cm_normalized, annot=True, xticklabels=label_list, yticklabels=label_list, fmt=".2f", cmap="Blues")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Confusion Matrix (Accuracy %)")
plt.tight_layout()
plt.show()
