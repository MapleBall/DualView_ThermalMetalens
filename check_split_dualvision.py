import torch
import torchvision.transforms.functional as TF
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt

class DualViewCrop(object):
    def __init__(self, crop_size=(128, 128), offset=(0, 0)):
        self.crop_size = crop_size
        self.offset = offset
        self.to_gray = transforms.Grayscale(num_output_channels=1)

    def __call__(self, img):
        # 假設原始圖像是左右排列的雙視野圖像
        w, h = img.size
        mid = w // 2

        # 使用 center_crop 直接在左右兩側進行裁剪
        left_crop = TF.center_crop(img.crop((0, 0, mid, h)), self.crop_size)
        
        # 右視野裁剪（帶偏移）
        right_start = mid + self.offset[0]
        right_crop = TF.center_crop(img.crop((right_start, self.offset[1], w, h)), self.crop_size)
        
        # 使用 Grayscale 轉換為灰度圖
        left_crop_gray = self.to_gray(left_crop)
        right_crop_gray = self.to_gray(right_crop)
        
        return left_crop_gray, right_crop_gray


# 載入測試圖像
try:
    test_img = Image.open(r'ThermalHand\ThermalGesture\1504.bmp')
except:
    print("無法載入 test.bmp，使用範例圖像替代")
    # 創建一個示例圖像用於測試
    import numpy as np
    # 創建一個左右分開的測試圖像
    test_array = np.zeros((200, 400), dtype=np.uint8)
    # 左側填充一個圓形
    for i in range(200):
        for j in range(200):
            if (i-100)**2 + (j-100)**2 < 80**2:
                test_array[i, j] = 255
    # 右側填充一個方形
    test_array[50:150, 250:350] = 255
    test_img = Image.fromarray(test_array)

# 定義圖像轉換
img_transform = DualViewCrop(crop_size=(194, 194), offset=(0, 0))

# 應用轉換
left_crop_gray, right_crop_gray = img_transform(test_img)

# 顯示結果
plt.figure(figsize=(10, 5))

plt.subplot(1, 3, 1)
plt.imshow(test_img, cmap='gray', vmin=0, vmax=255)
plt.title('Original Image')
plt.axis('off')

plt.subplot(1, 3, 2)
plt.imshow(left_crop_gray, cmap='gray', vmin=0, vmax=255)
plt.title('Left Crop')
plt.axis('off')

plt.subplot(1, 3, 3)
plt.imshow(right_crop_gray, cmap='gray', vmin=0, vmax=255)
plt.title('Right Crop')
plt.axis('off')

plt.tight_layout()
plt.show()

# 如果你想要將結果轉換為張量
left_tensor = TF.to_tensor(left_crop_gray)
right_tensor = TF.to_tensor(right_crop_gray)
combined_tensor = torch.cat([left_tensor, right_tensor], dim=0)
print("Combined tensor shape:", combined_tensor.shape)
