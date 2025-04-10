import cv2
import numpy as np
import os

def split_dual_view_image(image):
    """
    將影像水平分割為左視圖和右視圖
    """
    height, width = image.shape
    mid = width // 2
    left_view = image[:, :mid]
    right_view = image[:, mid:]
    return left_view, right_view

def process_images_in_folder(input_root_folder, output_root_folder):
    """
    遍歷 input_root_folder 下所有子資料夾並處理影像，將處理後的影像存入 output_root_folder 保持資料夾結構
    """
    image_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff')
    processed_count = 0

    for root, dirs, files in os.walk(input_root_folder):
        for file in files:
            if file.lower().endswith(image_extensions):
                input_path = os.path.join(root, file)
                
                # 計算輸出子資料夾路徑
                relative_dir = os.path.relpath(root, input_root_folder)
                output_subdir = os.path.join(output_root_folder, relative_dir)
                
                # 確保輸出資料夾存在
                os.makedirs(output_subdir, exist_ok=True)

                name, ext = os.path.splitext(file)

                try:
                    image = cv2.imread(input_path, cv2.IMREAD_GRAYSCALE)

                    if image is None:
                        print(f"無法讀取影像: {input_path}")
                        continue

                    left_view, right_view = split_dual_view_image(image)

                    left_output_path = os.path.join(output_subdir, f"{name}_left{ext}")
                    right_output_path = os.path.join(output_subdir, f"{name}_right{ext}")

                    cv2.imwrite(left_output_path, left_view)
                    cv2.imwrite(right_output_path, right_view)

                    print(f"已處理: {input_path}")
                    processed_count += 1

                except Exception as e:
                    print(f"處理 {input_path} 時發生錯誤: {str(e)}")

    print(f"✅ 完成：共處理 {processed_count} 張影像，已儲存至 {output_root_folder}")

# 使用範例:
process_images_in_folder('test', 'half_test_folder')
