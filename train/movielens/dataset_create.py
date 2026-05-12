from datasets import Dataset, Image, DatasetDict
from pathlib import Path
import pandas as pd
import os

os.environ['CURL_CA_BUNDLE'] = ''
os.environ["CUDA_VISIBLE_DEVICES"] = "1,2,3,4,5,6"


def get_file_full_paths_and_names(folder_path):
    folder_path = Path(folder_path)
    full_paths = []
    file_names = []
    for file_path in folder_path.glob('*'):
        if file_path.is_file():
            full_paths.append(str(file_path.absolute()))
            file_names.append(file_path.stem)  # 使用.stem获取不带扩展名的文件名
    return full_paths, file_names
# /home/team//MLLM-MSR-main/MLLM-MSR
train_pair_file_path = "/home/team//MLLM-MSR-main/MLLM-MSR/data/movielens/movielens_out/train_pairs.csv"
df_train = pd.read_csv(train_pair_file_path)
df_train['item'] = df_train['item'].astype(str)
df_train['user'] = df_train['user'].astype(str)

val_pair_file_path = "/home/team//MLLM-MSR-main/MLLM-MSR/data/movielens/movielens_out/val_pairs.csv"
df_val = pd.read_csv(val_pair_file_path)
df_val['item'] = df_val['item'].astype(str)
df_val['user'] = df_val['user'].astype(str)



# user_pref_file_path = "/home/team//MLLM-MSR-main/MLLM-MSR/Inference/movielens/user_preference_recurrent_movie.csv"
user_pref_file_path = "/home/team//MLLM-MSR-main/MLLM-MSR/Inference/movielens/user_preference_LS_movie_numeric.csv"
user_pref_df = pd.read_csv(user_pref_file_path, header=None, names=["user", "preference"])
user_pref_df['user'] = user_pref_df['user'].astype(str)
# "/home/team//MicroLens-50k-Dataset/MicroLens-50k_titles.csv"

item_title_file_path = "/home/team//movielens/ml-latest-small/movies.csv"
item_title_df = pd.read_csv(item_title_file_path, header=None, names=["item", "title","genres"])
item_title_df['item'] = item_title_df['item'].astype(str)


folder_path = "/home/team//movielens/poster"
file_paths, file_names = get_file_full_paths_and_names(folder_path)
image_df = pd.DataFrame({"image": file_paths, "item": file_names})
image_df['item'] = image_df['item'].astype(str)


df_train = pd.merge(df_train, image_df, on="item")
df_train = pd.merge(df_train, item_title_df, on="item")
df_train = pd.merge(df_train, user_pref_df, on="user")

df_val = pd.merge(df_val, image_df, on="item")
df_val = pd.merge(df_val, item_title_df, on="item")
df_val = pd.merge(df_val, user_pref_df, on="user")

# reasons_file = "/home/team//MLLM-MSR-main/MLLM-MSR/deepseek/movielens/movie_train_pairs_with_reasons_merged.csv"
reasons_file = "/home/team//MLLM-MSR-main/MLLM-MSR/deepseek/movielens/movie_train_pairs_with_reasons_numeric_merged.csv"
reasons_df = pd.read_csv(reasons_file)
reasons_df['user'] = reasons_df['user'].astype(str)
reasons_df['item'] = reasons_df['item'].astype(str)

# 保留合并需要的列
reasons_df = reasons_df[['user', 'item', 'label', 'answer']]

# 合并 train / val
df_train = pd.merge(df_train, reasons_df, on=['user', 'item', 'label'], how="left")


# prompt_text = "Based on the previous interaction history, the user's preference can be summarized as: {}" \
#               "Please predict whether this user would interact with the video at the next opportunity. The video's title is'{}', and the given image is this video's cover? " \
#               "Please only response 'yes' or 'no' based on your judgement, do not include any other content including words, space, and punctuations in your response."

# prompt_text = "As a vision-llm, your task involves analyzing a given video's cover image and title, alongside a summary of a user's preferences based on their interaction history. Respond with 'yes' or 'no' to indicate whether the user will interact with the video at their next opportunity. Please limit your response to only 'yes' or 'no', without including any additional content, words, or punctuation." \
#              "User's summarized preferences based on past interactions: {}" \
#              "Will the user interact with the video titled '{}' and represented by the above given cover image at the next opportunity? "

# prompt_text = (
#     "As a vision-llm, your task involves analyzing a given video's cover image and title, "
#     "alongside a summary of a user's preferences based on their interaction history. "
#     "Respond with 'yes' or 'no' to indicate whether the user will interact with the video at their next opportunity, "
#     "and provide a short explanation for your choice, describing why this recommendation aligns or does not align "
#     "with the user's long-term and short-term interests, or their preference dynamics. "
#     "User's summarized preferences based on past interactions: {}\n"
#     "Will the user interact with the video titled '{}', genre is {} and represented by the above given cover image at the next opportunity? "
#     "Please answer in the following format:\n"
#     "[ANS] yes/no\n"
#     "[REASON] short paragraph explaining the reason for this choice."
# )

prompt_text = (
    "As a vision-language model, your task is to analyze the given movie's cover image and title, "
    "together with the user's summarized preferences. "
    "The user CLICKED on this movie. Explain why, focusing on how the movie's content aligns with their preferences.\n\n"
    "User's summarized preferences:\n{}\n\n"
    "movie title: {}\n"
    "Please answer in the following structured format:\n"
    "[ANS] yes\n"
    "[REASON]\n"
    "【Long-term Alignment】List 1–3 concise points explaining how the movie matches the user's stable, long-term interests.\n"
    "【Short-term Alignment】List 1–3 concise points showing how it fits the user's current or recent focus.\n"
    "【Preference Dynamics】List 1–2 concise points describing how this click reflects the user's changing or exploratory behavior.\n"
    "Each point should be a short, factual statement (under 25 words). Avoid vague or repetitive expressions."
)
df_train['prompt'] = df_train.apply(lambda x: prompt_text.format(x['preference'], x['title'], x['genres']), axis=1)
df_train['ground_truth'] = df_train.apply(lambda x: 'Yes' if x['label'] == 1 else 'No', axis=1)
df_train['truth_reason'] = df_train['answer']
df_train = df_train[['prompt', 'image', 'ground_truth','truth_reason']]

df_val['prompt'] = df_val.apply(lambda x: prompt_text.format(x['preference'], x['title'], x['genres']), axis=1)
df_val['ground_truth'] = df_val.apply(lambda x: 'Yes' if x['label'] == 1 else 'No', axis=1)
df_val['truth_reason'] = ""
df_val = df_val[['prompt', 'image', 'ground_truth', 'truth_reason']]


train_dataset = Dataset.from_pandas(df_train)
# train_dataset = train_dataset.cast_column("image", Image())
# train_dataset = train_dataset.select(range(25000))
train_dataset = train_dataset.shuffle(seed=2024)

val_dataset = Dataset.from_pandas(df_val)
# val_dataset = val_dataset.cast_column("image", Image())
# val_dataset = val_dataset.select(range(1000))

dataset = DatasetDict({"train": train_dataset, "validation": val_dataset})

# dataset.save_to_disk("MicroLens-50k-training-recurrent-noLS-reason")
# dataset.save_to_disk("movielens-training-recurrent-noLS-yesno")
# dataset.save_to_disk("movielens-training-recurrent-longshortpreference-yesno")
dataset.save_to_disk("movielens-training-recurrent-longshortpreference-reason-numeric-imagepath")
