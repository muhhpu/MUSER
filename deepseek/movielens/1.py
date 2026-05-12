import os
import sys
import pandas as pd
from tqdm import tqdm
from openai import OpenAI
import numpy as np

# -----------------------------
# 1. 配置 DeepSeek
# -----------------------------
DEEPSEEK_API_KEY = "sk-05c1f11f21344176bd97b38847dc35bb"  # TODO: 替换成你的真实 key
client = OpenAI(api_key=DEEPSEEK_API_KEY, base_url="https://api.deepseek.com")

# -----------------------------
# 2. 命令行参数
# -----------------------------
if len(sys.argv) < 2:
    print("Usage: python 1.py <group_id>")
    sys.exit(1)

group_id = int(sys.argv[1])  # 0~9

# -----------------------------
# 3. 数据文件路径
# -----------------------------
# train_pairs_path = "/Users//Desktop/第三篇/MLLM-MSR-main/MLLM-MSR/data/microlens/preprocessing/train_pairs.csv"
# user_pref_file_path = "/Users//Desktop/第三篇/MLLM-MSR-main/MLLM-MSR/Inference/microlens/store/user_preference_recurrent_whole2.csv"
# item_title_file_path = "/Users//Desktop/第三篇/MLLM-MSR-main/MLLM-MSR/data/microlens/MicroLens-50k_titles.csv"
# image_summary_file_path = "/Users//Desktop/第三篇/MLLM-MSR-main/MLLM-MSR/image_summary.csv"
# train_pairs_path = "/home/team//MLLM-MSR-main/MLLM-MSR/data/movielens/movielens_out/train_pairs.csv"
train_pairs_path = "/home/team//MLLM-MSR-main/MLLM-MSR/data/movielens/movielens_out/test_pairs_filtered.csv"
# user_pref_file_path = "/home/team//MLLM-MSR-main/MLLM-MSR/Inference/microlens/user_preference_recurrent_whole2.csv"
user_pref_file_path = "/home/team//MLLM-MSR-main/MLLM-MSR/Inference/movielens/user_preference_LS_movie_numeric.csv"
item_title_file_path = "/home/team//movielens/ml-latest-small/movies.csv"
image_summary_file_path = "/home/team//MLLM-MSR-main/MLLM-MSR/movie_image_summary.csv"

# -----------------------------
# 4. 读取 CSV 数据
# -----------------------------
df = pd.read_csv(train_pairs_path)
df["user"] = df["user"].astype(str)
df["item"] = df["item"].astype(str)

user_pref_df = pd.read_csv(user_pref_file_path, header=None, names=["user", "preference"])
user_pref_df["user"] = user_pref_df["user"].astype(str)

item_title_df = pd.read_csv(item_title_file_path, header=None, names=["item", "title","genres"])
item_title_df["item"] = item_title_df["item"].astype(str)

image_summary_df = pd.read_csv(image_summary_file_path)
image_summary_df["item_id"] = image_summary_df["item_id"].astype(str)

# -----------------------------
# 5. 切分数据
# -----------------------------
df_splits = np.array_split(df, 2)  # 均匀分成 10 份
df = df_splits[group_id].reset_index(drop=True)


# -----------------------------
# 6. 输出 CSV 文件
# -----------------------------
output_file = f"./movie_test_pairs_with_reasons_numeric_{group_id}.csv"
os.makedirs(os.path.dirname(output_file), exist_ok=True)

# -----------------------------
# 7. prompt 模板
# -----------------------------
prompt_template_clicked = (
    "As a vision-language model, your task is to analyze the given video's cover image and title, "
    "together with the user's summarized preferences. "
    "The user CLICKED on this video. Explain why, focusing on how the video's content aligns with their preferences.\n\n"
    "User's summarized preferences:\n{preference}\n\n"
    "Video title: {title}\n"
    "Video cover description: {summary}\n\n"
    "Please answer in the following structured format:\n"
    "[ANS] yes\n"
    "[REASON]\n"
    "【Long-term Alignment】List 1–3 concise points explaining how the video matches the user's stable, long-term interests.\n"
    "【Short-term Alignment】List 1–3 concise points showing how it fits the user's current or recent focus.\n"
    "【Preference Dynamics】List 1–2 concise points describing how this click reflects the user's changing or exploratory behavior.\n"
    "Each point should be a short, factual statement (under 25 words). Avoid vague or repetitive expressions."
)
prompt_template_not_clicked = (
    "As a vision-language model, your task is to analyze the given video's cover image and title, "
    "together with the user's summarized preferences. "
    "The user did NOT click on this video. Explain why, focusing on how the video's content fails to align with their preferences.\n\n"
    "User's summarized preferences:\n{preference}\n\n"
    "Video title: {title}\n"
    "Video cover description: {summary}\n\n"
    "Please answer in the following structured format:\n"
    "[ANS] no\n"
    "[REASON]\n"
    "【Long-term Misalignment】List 1–3 concise points explaining how the video differs from the user's stable, long-term interests.\n"
    "【Short-term Misalignment】List 1–3 concise points showing how it contrasts with the user's current or recent focus.\n"
    "【Preference Dynamics】List 1–2 concise points describing how this non-click reflects the user's stability or lack of interest in new topics.\n"
    "Each point should be a short, factual statement (under 25 words). Avoid vague or repetitive expressions."
)
# prompt_template_clicked = (
#     "As a vision-llm, your task involves analyzing a given Movie's cover image and title and genres, "
#     "alongside a summary of a user's preferences based on their interaction history. "
#     "The user CLICKED on this Movie. Please explain why the user might have clicked it, "
#     "considering how the Movie's content aligns with their long-term interests, short-term interests, "
#     "or preference dynamics.\n\n"
#     "User's summarized preferences based on past interactions: {preference}\n\n"
#     "Movie title: {title}\n"
#     "Movie Genres: {genres}\n"
#     "Movie cover description: {summary}\n\n"
#     "Please answer in the following format:\n"
#     "[ANS] yes\n"
#     "[REASON] short paragraph explaining the reason for this choice."
# )
#
# prompt_template_not_clicked = (
#     "As a vision-llm, your task involves analyzing a given Movie's cover image and title, "
#     "alongside a summary of a user's preferences based on their interaction history. "
#     "The user did NOT click on this Movie. Please explain why the user might have ignored it, "
#     "considering how the Movie's content does not align with their long-term interests, short-term interests, "
#     "or preference dynamics.\n\n"
#     "User's summarized preferences based on past interactions: {preference}\n\n"
#     "Movie title: {title}\n"
#     "Movie Genres: {genres}\n"
#     "Movie cover description: {summary}\n\n"
#     "Please answer in the following format:\n"
#     "[ANS] no\n"
#     "[REASON] short paragraph explaining the reason for this choice."
# )

# -----------------------------
# 8. 遍历数据生成 DeepSeek 回答
# -----------------------------
records = []

for idx, row in tqdm(df.iterrows(), total=len(df)):
    user_id = row["user"]
    item_id = row["item"]
    label = row["label"]

    # 找用户偏好
    user_pref = user_pref_df[user_pref_df["user"] == user_id]["preference"]
    if user_pref.empty:
        user_pref = "No detailed user preference available."
    else:
        user_pref = user_pref.values[0]

    # 找 item title
    item_title = item_title_df[item_title_df["item"] == item_id]["title"]
    if item_title.empty:
        item_title = "Unknown title"
    else:
        item_title = item_title.values[0]

    item_genres = item_title_df[item_title_df["item"] == item_id]["genres"]

    if item_genres.empty:
        item_genres = "Unknown title"
    else:
        item_genres = item_genres.values[0]

    # 找图像 summary
    item_summary = image_summary_df[image_summary_df["item_id"] == item_id]["summary"]
    if item_summary.empty:
        item_summary = "No cover description available."
    else:
        item_summary = item_summary.values[0]


    # 构造 prompt
    if label == 1:
        prompt = prompt_template_clicked.format(
            preference=user_pref,
            title=item_title,
            summary=item_summary,
            genres=item_genres
        )
    else:
        prompt = prompt_template_not_clicked.format(
            preference=user_pref,
            title=item_title,
            summary=item_summary,
            genres=item_genres
        )

    try:
        response = client.chat.completions.create(
            model="deepseek-chat",
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt},
            ],
            stream=False
        )

        generated_text = response.choices[0].message.content.strip()

    except Exception as e:
        print(f"Error at sample {idx}: {e}")
        generated_text = "ERROR"

    # 保存到结果列表
    records.append({
        "user": user_id,
        "item": item_id,
        "label": label,
        "answer": generated_text
    })

# -----------------------------
# 9. 保存 CSV
# -----------------------------
result_df = pd.DataFrame(records)
result_df.to_csv(output_file, index=False, encoding="utf-8")

print(f"✅ Done. Group {group_id} results saved to {output_file}")
print(result_df.head())
