import pandas as pd

# 读两个 csv
file1 = "user_preference_recurrent_whole1.csv"
file2 = "user_preference_recurrent_whole2.csv"

df1 = pd.read_csv(file1)
df2 = pd.read_csv(file2)

# 合并（按行拼接）
df = pd.concat([df1, df2], ignore_index=True)

# 如果同一个 user_id 在两个文件里都出现过，可以去重
df = df.drop_duplicates(subset=["user_id"])

# 保存到新文件
df.to_csv("user_preference_longshort.csv", index=False)

print("合并完成，结果保存在 user_summary_merged.csv")
