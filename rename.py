import os
import glob

def rename_bmp_files(folder_path, start_index=1798):
    # 獲取資料夾中所有的bmp檔案
    bmp_files = glob.glob(os.path.join(folder_path, "*.bmp"))
    
    # 排序檔案列表，確保命名順序一致
    bmp_files.sort()
    
    # 重新命名檔案
    for index, file_path in enumerate(bmp_files, start=start_index):
        # 創建新的檔案名稱，格式為"0149.bmp", "0150.bmp"等
        new_filename = f"{index:04d}.bmp"
        new_file_path = os.path.join(folder_path, new_filename)
        
        # # 如果新檔案名稱已存在但不是當前處理的檔案，先將其重命名為臨時檔案
        # if os.path.exists(new_file_path) and new_file_path != file_path:
        #     temp_path = os.path.join(folder_path, f"temp_{index:04d}.bmp")
        #     os.rename(file_path, temp_path)
        #     file_path = temp_path
        
        # 重新命名檔案
        os.rename(file_path, new_file_path)
        print(f"已將檔案重命名為: {new_filename}")

# 使用範例
if __name__ == "__main__":
    folder_path = r"ThermalGesture\love"
    if os.path.isdir(folder_path):
        rename_bmp_files(folder_path)
        print("所有BMP檔案重命名完成！")
    else:
        print("無效的資料夾路徑，請確認後重試。")
