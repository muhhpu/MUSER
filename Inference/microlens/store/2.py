import pandas as pd
import re

# === 读取 CSV ===
df = pd.read_csv("user_preference_recurrent_numeric.csv", encoding="utf-8")

# === 目标关键词 ===
kw1 = "basketball"
kw2 = "mobile game"

# === 存放符合条件的用户 ===
qualified_users = []

# === 遍历每一行 ===
for _, row in df.iterrows():
    user_id = row["user_id"]
    summary = row["summary"]

    # 提取各部分内容（Long-term, Short-term, Preference Dynamics）
    long_term_match = re.search(r"【Long-term Interests】(.*?)【", summary, flags=re.S)
    short_term_match = re.search(r"【Short-term Interests】(.*?)【", summary, flags=re.S)

    long_term = long_term_match.group(1).lower() if long_term_match else ""
    short_term = short_term_match.group(1).lower() if short_term_match else ""

    # 检查关键词出现位置
    has_kw1_long = kw1.lower() in long_term
    has_kw2_long = kw2.lower() in long_term
    has_kw1_short = kw1.lower() in short_term
    has_kw2_short = kw2.lower() in short_term

    # 条件：两个关键词必须都出现，且不在同一个 term 内
    if (has_kw1_long and has_kw2_short) or (has_kw1_short and has_kw2_long):
        qualified_users.append(user_id)

# === 输出结果 ===
print("✅ 符合条件的用户 ID:")
print(qualified_users)