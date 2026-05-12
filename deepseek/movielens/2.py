import pandas as pd
import glob

# 匹配所有 train_pairs_with_reasons_*.csv
file_list = sorted(glob.glob("movie_test_pairs_with_reasons_numeric_*.csv"))

print("发现的文件：")
for f in file_list:
    print(f)

# 读取并合并
dfs = [pd.read_csv(f) for f in file_list]
merged_df = pd.concat(dfs, ignore_index=True)

# 保存
output_file = "movie_test_pairs_with_reasons_numeric_merged.csv"
merged_df.to_csv(output_file, index=False, encoding="utf-8")

print(f"✅ 合并完成，共 {len(merged_df)} 行，保存到 {output_file}")