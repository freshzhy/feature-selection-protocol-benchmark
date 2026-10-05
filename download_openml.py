import openml
import pandas as pd
import os

# 保存目录，自己指定
SAVE_DIR = "~/Downloads/dataset/openml"
os.makedirs(SAVE_DIR, exist_ok=True)

dataset_ids = [37, 31, 15, 29, 3, 1462, 1464, 1489, 1494, 
               40982, 40975, 1590, 1063, 1067]

for did in dataset_ids:
    try:
        ds = openml.datasets.get_dataset(did)
        X, y, _, feat_names = ds.get_data(
            target=ds.default_target_attribute,
            dataset_format='dataframe'
        )
        # 合并 X 和 y，保存为 CSV
        df = X.copy()
        df['target'] = y
        save_path = os.path.join(SAVE_DIR, f"{ds.name}.csv")
        df.to_csv(save_path, index=False)
        print(f"✓ {ds.name}: {X.shape[0]} 样本, {X.shape[1]} 特征 → 已保存至 {save_path}")
    except Exception as e:
        print(f"✗ ID {did}: {e}")