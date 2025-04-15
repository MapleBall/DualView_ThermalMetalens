import pandas as pd
import numpy as np
import os

# 定義類別種類
classes = ['zero', 'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine', 'like', 'ok', 'love']

# 設定資料夾路徑
parent_folder = "test_2"        # 母資料夾，存放影像
sub_csv_folder = "csv_data"    # 子 CSV 存放位置
csv_filename = "test_v2.csv"      # 最終合併輸出的 CSV 檔案名稱

# 讀取子 CSV 作為主要資料來源
sub_csv_dfs = []
if os.path.exists(sub_csv_folder):
    for file in os.listdir(sub_csv_folder):
        if file.endswith(".csv"):
            file_path = os.path.join(sub_csv_folder, file)
            try:
                df = pd.read_csv(file_path)
                if not df.empty:
                    sub_csv_dfs.append(df)
                    print(f"已讀取子 CSV：{file}")
            except Exception as e:
                print(f"⚠️ 讀取子 CSV '{file}' 時出錯：{e}")

# 合併所有子 CSV（可為空）
if sub_csv_dfs:
    merged_sub_csv_df = pd.concat(sub_csv_dfs, ignore_index=True)
    print(f"已合併 {len(sub_csv_dfs)} 個子 CSV，共 {len(merged_sub_csv_df)} 筆資料")
else:
    merged_sub_csv_df = pd.DataFrame(columns=["image_number"] + classes)
    print("⚠️ 沒有可用的子 CSV 資料")

# 接著處理母資料夾影像資料
x = []
y = []
idx_start = 2631

if os.path.exists(parent_folder):
    for folder_name in classes:
        folder_path = os.path.join(parent_folder, folder_name)
        if os.path.exists(folder_path):
            img_names = [f for f in os.listdir(folder_path) if f.endswith('.bmp')]
            folder_N = len(img_names)
            if folder_N > 0:
                print(f"處理 '{folder_name}' 資料夾中的 {folder_N} 個影像")
                one_hot = np.eye(len(classes))[classes.index(folder_name)]
                image_numbers = [f"{i:04d}" for i in range(idx_start, idx_start + folder_N)]
                x.extend(image_numbers)
                y.extend([one_hot] * folder_N)
                idx_start += folder_N

# 建立母資料夾轉換後的新資料
new_csv_data = [[img_num] + list(label) for img_num, label in zip(x, y)]
new_csv_df = pd.DataFrame(new_csv_data, columns=["image_number"] + classes)

# 最終合併：子 CSV 資料（在前） + 新產生資料（在後）
final_df = pd.concat([merged_sub_csv_df, new_csv_df], ignore_index=True)

# 去除重複 image_number（以後者為主，保留最新）
final_df.drop_duplicates(subset=["image_number"], keep="last", inplace=True)

# 儲存為 CSV
if not final_df.empty:
    final_df.to_csv(csv_filename, index=False)
    print(f"✅ 已儲存最終合併資料，共 {len(final_df)} 筆至 '{csv_filename}'")
else:
    print("⚠️ 沒有任何資料儲存")
