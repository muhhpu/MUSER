from datasets import Dataset, Image
from pathlib import Path
import pandas as pd
import os

os.environ['CURL_CA_BUNDLE'] = ''
os.environ["CUDA_VISIBLE_DEVICES"] = "1,2,3,4,6,7"


def get_file_full_paths_and_names(folder_path):
    folder_path = Path(folder_path)
    full_paths = []
    file_names = []
    for file_path in folder_path.glob('*'):
        if file_path.is_file():
            full_paths.append(str(file_path.absolute()))
            file_names.append(file_path.stem)  # 使用.stem获取不带扩展名的文件名
    return full_paths, file_names

pair_file_path = "/home/team//MLLM-MSR-main/MLLM-MSR/data/movielens/movielens_out/test_pairs.csv"
# pair_file_path = "/home/team//MLLM-MSR-main/MLLM-MSR/data/movielens/movielens_out/test_pairs_filtered.csv"
df = pd.read_csv(pair_file_path)
df['item'] = df['item'].astype(str)
df['user'] = df['user'].astype(str)

# user_pref_file_path = "/home/team//MLLM-MSR-main/MLLM-MSR/user_preference_recurrent.csv"
# user_pref_file_path = "/home/team//MLLM-MSR-main/MLLM-MSR/Inference/microlens/store/user_preference_longshort.csv"
user_pref_file_path = "/home/team//MLLM-MSR-main/MLLM-MSR/Inference/movielens/user_preference_noRS_movie.csv"
# user_pref_file_path = "/home/team//MLLM-MSR-main/MLLM-MSR/Inference/movielens/user_preference_recurrent_movie.csv"
# user_pref_file_path = "/home/team//MLLM-MSR-main/MLLM-MSR/Inference/movielens/user_preference_LS_movie_numeric.csv"
user_pref_df = pd.read_csv(user_pref_file_path, header=None, names=["user", "preference"])
user_pref_df['user'] = user_pref_df['user'].astype(str)


item_title_file_path = "/home/team//movielens/ml-latest-small/movies.csv"
item_title_df = pd.read_csv(item_title_file_path, header=None, names=["item", "title","genres"])
item_title_df['item'] = item_title_df['item'].astype(str)


folder_path = "/home/team//movielens/poster"
file_paths, file_names = get_file_full_paths_and_names(folder_path)
image_df = pd.DataFrame({"image": file_paths, "item": file_names})
image_df['item'] = image_df['item'].astype(str)


df = pd.merge(df, image_df, on="item")
df = pd.merge(df, item_title_df, on="item")
df = pd.merge(df, user_pref_df, on="user")


reasons_file = "/home/team//MLLM-MSR-main/MLLM-MSR/deepseek/movielens/movie_test_pairs_with_reasons_numeric_merged.csv"
reasons_df = pd.read_csv(reasons_file)
reasons_df['user'] = reasons_df['user'].astype(str)
reasons_df['item'] = reasons_df['item'].astype(str)
# 保留合并需要的列
reasons_df = reasons_df[['user', 'item', 'label', 'answer']]
# 合并 train / val
df = pd.merge(df, reasons_df, on=['user', 'item', 'label'], how="left")

prompt_text = "[INST]<image>\n As a vision-llm, you will be given the cover image and the title of a video and the summarized preference of a user, and your task is to predict whether the user would interact with the video. Please only response 'yes' or 'no' based on your judgement, do not include any other content including words, space, and punctuations in your response.\n " \
             "Based on the previous interaction history, the user's preference can be summarized as: {}" \
        "Please predict whether this user would interact with the video at the next opportunity. The video's title is'{}', and the given image is this video's cover? [/INST]"

# prompt_text = "Based on the previous interaction history, the user's preference can be summarized as: {}" \
#               "Please predict whether this user would interact with the video at the next opportunity. The video's title is'{}', and the given image is this video's cover? " \
#               "Please only response 'yes' or 'no' based on your judgement, do not include any other content including words, space, and punctuations in your response."


# prompt_text = "[INST] <image>\n As a vision-llm, your task involves analyzing a video's given cover image and title, alongside a summary of a user's preferences based on their interaction history. Respond with 'yes' or 'no' to indicate whether the user will interact with the video at their next opportunity. Please limit your response to only 'yes' or 'no', without including any additional content, words, or punctuation." \
#              "User's summarized preferences based on past interactions: {}" \
#              "Will the user interact with the video titled '{}' and represented by the above given cover image at the next opportunity? [/INST]"
# prompt_text = (
#     "[INST] <image>\n"
#     "As a vision-llm, your task involves analyzing a given video's cover image and title, "
#     "alongside a summary of a user's preferences based on their interaction history. "
#     "Respond with 'yes' or 'no' to indicate whether the user will interact with the video at their next opportunity, "
#     "and provide a short explanation for your choice, describing why this recommendation aligns or does not align "
#     "with the user's long-term and short-term interests, or their preference dynamics. "
#     "User's summarized preferences based on past interactions: {}\n"
#     "Will the user interact with the video titled '{}', genre is {}  and represented by the above given cover image at the next opportunity? "
#     "Please answer in the following format:\n"
#     "[ANS] yes/no\n"
#     "[REASON] short paragraph explaining the reason for this choice. [/INST]"
# )
# prompt_text = (
#     "[INST] <image>\n"
#     "As a vision-language model, your task is to analyze the given video's cover image and title, "
#     "together with the user's summarized preferences. "
#     "The user CLICKED on this video. Explain why, focusing on how the video's content aligns with their preferences.\n\n"
#     "User's summarized preferences:\n{}\n\n"
#     "Video title: {}\n"
#     "Please answer in the following structured format:\n"
#     "[ANS] yes\n"
#     "[REASON]\n"
#     "【Long-term Alignment】List 1–3 concise points explaining how the video matches the user's stable, long-term interests.\n"
#     "【Short-term Alignment】List 1–3 concise points showing how it fits the user's current or recent focus.\n"
#     "【Preference Dynamics】List 1–2 concise points describing how this click reflects the user's changing or exploratory behavior.\n"
#     "Each point should be a short, factual statement (under 25 words). Avoid vague or repetitive expressions.[/INST]"
# )
df['prompt'] = df.apply(lambda x: prompt_text.format(x['preference'], x['title'], x['genres']), axis=1)

# df = df[['user', 'prompt', 'image', 'label','answer']]
df = df[['user', 'prompt', 'image', 'label']]
#print(df.head())

# 创建数据集并指定列类型
#dataset = Dataset.from_dict({"image": file_paths, "item": file_names})
dataset = Dataset.from_pandas(df)
dataset = dataset.cast_column("image", Image())
#dataset = dataset.select(range(2000))

# 检查数据集结构
print(dataset)

dataset.save_to_disk("movielens-test-recurrent-noLS-yesno")
# dataset.save_to_disk("movielens-test-recurrent-longshortpreference-reason-numeric-testreason")

