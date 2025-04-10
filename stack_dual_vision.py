import cv2
import numpy as np

def stack_left_and_right_views(left_view, right_view):
    """
    將二維左右視野影像堆疊成三維張量

    Returns:
        stacked_tensor: A tensor with shape [height, width, channel=2].
    """
    # 確保左右影像具有相同的尺寸
    min_height = min(left_view.shape[0], right_view.shape[0])
    min_width = min(left_view.shape[1], right_view.shape[1])
    
    # 裁剪左右影像到相同的大小
    left_view = left_view[:min_height, :min_width]
    right_view = right_view[:min_height, :min_width]
    
    # 添加通道維度，將影像從 2D (height, width) 轉換為 3D (height, width, channel=1)
    left_channel = np.expand_dims(left_view, axis=-1)  # [height, width, 1]
    right_channel = np.expand_dims(right_view, axis=-1)  # [height, width, 1]
    
    # 堆疊左右影像，形成 [height, width, channel=2]
    stacked_tensor = np.concatenate([left_channel, right_channel], axis=-1)
    
    return stacked_tensor

# 使用範例
if __name__ == "__main__":
    # 假設有兩個灰度影像：左視野和右視野
    left_view = cv2.imread("Snapshot_426_left.bmp", cv2.IMREAD_GRAYSCALE)
    right_view = cv2.imread("Snapshot_426_right.bmp", cv2.IMREAD_GRAYSCALE)
    
    if left_view is None or right_view is None:
        print("無法讀取影像")
        exit()
    
    # 堆疊成張量
    stacked_tensor = stack_left_and_right_views(left_view, right_view)
    
    print("堆疊後的張量形狀:", stacked_tensor.shape)  # 應為 [height, width, channel=2]
