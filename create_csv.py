import pandas as pd
import numpy as np

# 定義類別種類
classes = ['zero', 'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine', 'like', 'ok', 'love']

# 讀取之前創建的CSV檔案
csv_filename = "image_labels_combined.csv"
csv_df = pd.read_csv(csv_filename)
# dataset_x_path = []
# y = []
# idx_start = 0
# idx_end = floder_N
# for folder_name in classes
# floder_N = len(img_names)
# x.appand(new_image_number)
# y.appand(one_hot)
# idx_start = idx_end +1
# idx_end = idx_start + floder_N
# 假設我們要添加編號0150~0299的圖片，標籤為"one"
new_image_numbers = [f"{i:04d}" for i in range(1798, 1948)]

# 創建"one"類別的one-hot編碼
one_one_hot = np.eye(len(classes))[classes.index('love')]

# 創建新的CSV資料
new_csv_data = []
for img_num in new_image_numbers:
    row = [img_num] + list(one_one_hot)
    new_csv_data.append(row)

# 創建新的DataFrame
columns = ["image_number"] + classes
new_csv_df = pd.DataFrame(new_csv_data, columns=columns)

# 合併原有的DataFrame和新的DataFrame
combined_csv_df = pd.concat([csv_df, new_csv_df], ignore_index=True)

# 儲存合併後的CSV檔案
combined_csv_filename = "image_labels_combined.csv"
combined_csv_df.to_csv(combined_csv_filename, index=False)

