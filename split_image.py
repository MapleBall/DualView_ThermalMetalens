import cv2
import numpy as np
import os

def find_center(image):
    """
    Find the center of a hand in a grayscale image.
    
    Args:
        image: Grayscale image containing a hand
        
    Returns:
        (cx, cy): Coordinates of the hand center, or (image center) if no hand is found
    """
    _, thresh = cv2.threshold(image, 135, 255, cv2.THRESH_BINARY) #二值化
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if contours:
        # Find the largest contour
        hand_contour = max(contours, key=cv2.contourArea)
        
        # Calculate the moments of the contour
        M = cv2.moments(hand_contour)
        
        # Calculate the centroid
        if M['m00'] != 0:
            cx = int(M['m10'] / M['m00'])
            cy = int(M['m01'] / M['m00'])
        else:
            # If m00 is zero, set the centroid to the image center
            cx = image.shape[1] // 2
            cy = image.shape[0] // 2
            
        return cx, cy
    else:
        # Unable to find hand region, use image center
        cx = image.shape[1] // 2
        cy = image.shape[0] // 2
        return cx, cy

def split_dual_view_image(image):
    """
    Split a dual-view image into left and right views centered on the hand centers.
    
    Args:
        image: Grayscale dual-view image
        
    Returns:
        left_view, right_view: The extracted left and right views
    """
    height, width = image.shape
    
    # Split the image in half for initial processing
    middle_x = width // 2
    left_half = image[:, :middle_x]
    right_half = image[:, middle_x:]
    
    # Find the center of the hand in each half
    left_cx, left_cy = find_center(left_half)
    right_cx, right_cy = find_center(right_half)
    
    # Adjust right_cx to account for the split
    right_cx += middle_x
    
    # Calculate the size of the views to extract
    # We'll use the minimum distance from the center to the edge
    view_size = min(
        left_cx,                
        middle_x - left_cx,     
        right_cx - middle_x,    
        width - right_cx,       
        left_cy,                
        height - left_cy,       
        right_cy,               
        height - right_cy       
    )
    
    # Ensure view_size is positive
    view_size = max(view_size, 10)
    
    # Extract equal-sized views centered on the detected hand centers
    left_view = image[max(0, left_cy-view_size):min(height, left_cy+view_size), 
                     max(0, left_cx-view_size):min(width, left_cx+view_size)]
    
    right_view = image[max(0, right_cy-view_size):min(height, right_cy+view_size), 
                      max(0, right_cx-view_size):min(width, right_cx+view_size)]
    
    return left_view, right_view

def stack_left_and_right_views(left_view, right_view):
    """
    將二維左右視野影像堆疊成三維張量
    Returns:
        stacked_tensor: 形狀為 [height, width, channel=2] 的張量
    """
    # 確保尺寸一致
    min_height = min(left_view.shape[0], right_view.shape[0])
    min_width = min(left_view.shape[1], right_view.shape[1])
    
    left_view = left_view[:min_height, :min_width]
    right_view = right_view[:min_height, :min_width]
    
    # 添加通道維度並堆疊
    stacked_tensor = np.concatenate([
        np.expand_dims(left_view, axis=-1),
        np.expand_dims(right_view, axis=-1)
    ], axis=-1)
    
    return stacked_tensor


def process_all_images(input_root_folder, output_root_folder):
    
    # 確保輸出根資料夾存在
    if not os.path.exists(output_root_folder):
        os.makedirs(output_root_folder)
    
    # 定義支援的影像副檔名
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff']
    
    # 遍歷所有資料夾和子資料夾
    for dirpath, dirnames, filenames in os.walk(input_root_folder):
        # 計算相對路徑，以便在輸出資料夾中創建相同的結構
        rel_path = os.path.relpath(dirpath, input_root_folder)
        output_path = os.path.join(output_root_folder, rel_path)
        
        # 確保對應的輸出資料夾存在
        if not os.path.exists(output_path):
            os.makedirs(output_path)
        
        # 處理當前資料夾中的所有影像檔案
        for filename in filenames:
            # 檢查副檔名是否為支援的影像格式
            _, ext = os.path.splitext(filename)
            if ext.lower() not in image_extensions:
                continue
            
            
            # 完整的檔案路徑
            file_path = os.path.join(dirpath, filename)
            # 檔案名稱（不含副檔名）
            name_without_ext = os.path.splitext(filename)[0]
            
            try:
                # 讀取影像
                image = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
                
                if image is None:
                    print(f"無法讀取影像: {file_path}")
                    continue
                
                # 分割影像
                left_view, right_view = split_dual_view_image(image)
                
                # 保存分割後的影像
                left_output_path = os.path.join(output_path, f"{name_without_ext}_left{ext}")
                right_output_path = os.path.join(output_path, f"{name_without_ext}_right{ext}")
                
                cv2.imwrite(left_output_path, left_view)
                cv2.imwrite(right_output_path, right_view)

                # 新增張量堆疊與保存
                stacked_tensor = stack_left_and_right_views(left_view, right_view)
                stack_output_path = os.path.join(output_path, f"{name_without_ext}_stack.npy")
                np.save(stack_output_path, stacked_tensor)  # 保存為.npy文件

            except Exception as e:
                print(f"處理 {file_path} 時發生錯誤: {str(e)}")
    
    print(f"處理結果已保存到 {output_root_folder}")

# 使用範例:
process_all_images('hand', 'prepocessed_hand')
