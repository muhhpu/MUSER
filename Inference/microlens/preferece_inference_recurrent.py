import torch
import transformers
from multiprocess import set_start_method
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import Dataset, load_dataset
import os
import pandas as pd
from torch.cuda.amp import autocast
import logging

os.environ['CURL_CA_BUNDLE'] = ''
os.environ["CUDA_VISIBLE_DEVICES"] = "1,2,3,4,5,6,7"

logging.getLogger("transformers").setLevel(logging.ERROR)

# model_id = "meta-llama/Meta-Llama-3-8B-Instruct"
# tokenizer = AutoTokenizer.from_pretrained(model_id, token='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')
# model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.bfloat16, attn_implementation="flash_attention_2", token='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').eval()

import os

model_path = os.path.abspath("./Meta-Llama-3-8B-Instruct")  # 转成绝对路径

tokenizer = AutoTokenizer.from_pretrained(
    model_path,
    local_files_only=True
)

model = AutoModelForCausalLM.from_pretrained(
    model_path,
    torch_dtype=torch.bfloat16,
    attn_implementation="flash_attention_2",
    local_files_only=True
).eval()

tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = 'left'
model.generation_config.pad_token_id = model.generation_config.eos_token_id

BATCH_SIZE = 12

terminators = [
    tokenizer.eos_token_id,
    tokenizer.convert_tokens_to_ids("<|eot_id|>")
]

pipelines = {}

def create_prompt(example, title_df, visual_df):
    user, items = example['user'], example['items']
    prompt = f"[INST] Below is a chronological list of videos previously watched by User {user}:\n"
    for i, item in enumerate(items):
        title = title_df.loc[item, 'title']
        visual_desc = visual_df.loc[item, 'summary']
        prompt += f"{i + 1}. {item}: Title - '{title}', Video cover description - {visual_desc}.\n"
    prompt += (
        "Based on the videos listed above, please summarize the user's preferences in terms of both content and visual style in one line. "
        "Only provide information about the user's preferences; do not repeat details about the previously watched videos. "
        "Do not repeat the question in your answer, and keep clear and concise."
        "The answer should start with 'The user appears to have a preference for'."
    )
    return {'prompt': prompt}


def map_prompt(example):
    return create_prompt(example, title_df, visual_df)

ui_pair_path = "/home/team//MLLM-MSR-main/MLLM-MSR/data/microlens/preprocessing/user_items_negs_all.tsv"
data = []
with open(ui_pair_path, 'r') as file:
    for line in file:
        parts = line.strip().split('\t')
        if len(parts) >= 2:
            user = parts[0]
            items = parts[1].split(' ')
            data.append({'user': user, 'items': items})
# # 找到最长的那一行
# max_len = -1
# max_idx = -1
# for i in range(len(data)):
#     if len(data[i]['items']) > max_len:
#         max_len = len(data[i]['items'])
#         max_idx = i
#
# print("最长序列长度:", max_len, "在索引:", max_idx)
# print(data[0])
# # 只保留这一行
# single_data = [data[max_idx]]
# single_data.append(data[0])
# print(single_data)



user_hist_df = pd.DataFrame(data)
user_hist_dataset = Dataset.from_pandas(user_hist_df)

title_df = pd.read_csv("/home/team//MicroLens-50k-Dataset/MicroLens-50k_titles.csv")
visual_df = pd.read_csv("/home/team//MLLM-MSR-main/MLLM-MSR/image_summary.csv")

title_df["item"] = title_df["item"].astype(str)
visual_df["item_id"] = visual_df["item_id"].astype(str)
title_df.set_index("item", inplace=True)
visual_df.set_index("item_id", inplace=True)


#  原
# def build_prompt(user, item_chunk, last_preference):
#     prompt = f"[INST] Below is a chronological list of videos previously watched by User {user}:\n"
#     for i, item in enumerate(item_chunk):
#         item = item.replace(',','')
#         title = title_df.loc[item, 'title']
#         visual_desc = visual_df.loc[item, 'summary']
#         prompt += f"{i + 1}. {item}: Title - '{title}', Video cover description - {visual_desc}.\n"
#     if last_preference is None:
#         prompt += "Based on the content and visual style of videos listed above, "
#     else:
#         prompt += f"We also know this user's previous preference.\n {last_preference}\n"
#         prompt += "Based on the content and visual style of videos listed above as well as the known aspects of user's preference, "
#     prompt += (
#         "please summarize the user's preferences in one continuous paragraph. "
#         "Only provide information about the user's preferences; do not repeat details about the previously watched videos. "
#         "Do not repeat the question in your answer, and keep clear and concise."
#         "The answer should start with 'The user appears to have a preference for'."
#     )
#     return prompt

# 第一种
# def build_prompt(user, item_chunk, last_preference):
#     prompt = f"[INST] Below is a chronological list of videos most recently watched by User {user}:\n"
#     for i, item in enumerate(item_chunk):
#         item = item.replace(',', '')
#         title = title_df.loc[item, 'title']
#         visual_desc = visual_df.loc[item, 'summary']
#         prompt += f"{i + 1}. {item}: Title - '{title}', Video cover description - {visual_desc}.\n"
#
#     if last_preference is None:
#         prompt += "\nNo prior preference summary is available.\n"
#     else:
#         prompt += f"\nWe also know this user's previously inferred preference summary:\n{last_preference}\n"
#
#     prompt += (
#         "\nYour task is to summarize the user’s preferences considering both their recent activity "
#         "(the videos listed above) and their longer-term viewing history (if available).\n"
#         "Please provide two aspects in your answer:\n"
#         "1. **Current short-term preference**: Describe the main themes, genres, or styles the user is currently showing interest in.\n"
#         "2. **Long-term preference and trend**: Describe the overall user preference across all known history, "
#         "and explain whether the user shows (a) broad and diverse interests with no strong bias, "
#         "(b) a clear and consistent preference for certain content, or (c) shifting interests over time "
#         "(e.g., moving from one type of content to another).\n\n"
#         "Only provide information about the user’s preferences and possible changes. "
#         "Do not repeat details of individual videos. "
#         "Keep the answer concise, and make it a continuous paragraph in natural language. "
#         "The answer should start with 'The user appears to have a preference for'."
#     )
#     return prompt

# # 第二种
# def build_prompt(user, item_chunk, last_preference):
#     prompt = f"[INST] Below is a chronological list of videos recently watched by User {user}:\n"
#     for i, item in enumerate(item_chunk):
#         item = item.replace(',', '')
#         title = title_df.loc[item, 'title']
#         visual_desc = visual_df.loc[item, 'summary']
#         prompt += f"{i + 1}. {item}: Title - '{title}', Video cover description - {visual_desc}.\n"
#
#     if last_preference is None:
#         prompt += (
#             "Please analyze this sequence of videos to summarize the user's preferences.\n"
#             "You should describe both:\n"
#             "1. **Long-term interests** (stable patterns or dominant themes suggested by the history).\n"
#             "2. **Short-term interests** (what seems to be emphasized in this batch of videos).\n"
#             "3. **Preference dynamics** (whether the user appears broad in interest, highly focused, or shows stage-based shifts).\n"
#             "Provide your answer in one coherent paragraph starting with:\n"
#             "'The user appears to have a preference for...'"
#         )
#     else:
#         prompt += (
#             f"We also know the following about the user's previous long-term preferences:\n"
#             f"{last_preference}\n\n"
#             "Based on both the above historical knowledge and this new batch of videos, "
#             "summarize the user's preferences with a focus on:\n"
#             "1. Updating the **long-term interest profile** if new evidence strengthens or contradicts prior trends.\n"
#             "2. Highlighting **short-term interests** shown in this batch.\n"
#             "3. Indicating **preference dynamics** (broad vs. narrow focus, stability vs. shifts, stage-based changes).\n"
#             "Write in one continuous paragraph, concise and analytical.\n"
#             "The answer should start with:\n"
#             "'The user appears to have a preference for...'"
#         )
#
#     return prompt
# 增强了可读性
# def build_prompt(user, item_chunk, last_preference):
#     prompt = f"[INST] Below is a chronological list of videos recently watched by User {user}:\n"
#     for i, item in enumerate(item_chunk):
#         item = item.replace(',', '')
#         title = title_df.loc[item, 'title']
#         visual_desc = visual_df.loc[item, 'summary']
#         prompt += f"{i + 1}. {item}: Title - '{title}', Video cover description - {visual_desc}.\n"
#
#     if last_preference is None:
#         prompt += (
#             "\nPlease analyze these videos and summarize the user's preferences in a structured format.\n"
#             "Your analysis must be divided into three clearly separated parts, each beginning with the given header:\n"
#             "【Long-term Interests】Describe stable, overarching themes (e.g., genres, topics, or aesthetics) inferred from the user's general viewing pattern.\n"
#             "【Short-term Interests】Describe the temporary or currently active focuses reflected in this batch (e.g., trends, new topics, or repeated keywords).\n"
#             "【Preference Dynamics】Describe how the user’s interests evolve — whether they stay consistent, shift toward new directions, or show exploratory tendencies.\n"
#             "Keep the language concise, analytical, and readable for humans. Each part should be 2–4 sentences.\n"
#             "Avoid redundancy and avoid generic phrases like 'the user likes diverse content'."
#         )
#     else:
#         prompt += (
#             f"\nThe user's previous preference profile was as follows:\n{last_preference}\n\n"
#             "Now, based on both the previous knowledge and the new batch of videos, update the profile.\n"
#             "For each of the following sections, revise and refine the prior understanding accordingly:\n"
#             "【Long-term Interests】Update stable trends if new evidence strengthens or contradicts prior themes.\n"
#             "【Short-term Interests】Summarize newly emerging or temporarily dominant interests in this batch.\n"
#             "【Preference Dynamics】Assess whether the user’s overall behavior shows stability, novelty seeking, or stage-based changes.\n"
#             "Each section should remain short (2–4 sentences), factual, and logically coherent.\n"
#             "Do not output in one paragraph — keep the section headers exactly as shown."
#         )
#
#     return prompt

# 增加了numeric
def build_prompt(user, item_chunk, last_preference):
    prompt = f"[INST] Below is a chronological list of videos recently watched by User {user}:\n"
    for i, item in enumerate(item_chunk):
        item = item.replace(',', '')
        title = title_df.loc[item, 'title']
        visual_desc = visual_df.loc[item, 'summary']
        prompt += f"{i + 1}. {item}: Title - '{title}', Video cover description - {visual_desc}.\n"

    if last_preference is None:
        prompt += (
            "\nPlease analyze these videos and summarize the user's preferences in a clear, structured, and concise format.\n"
            "The output must include the following three sections, each with numbered bullet points:\n"
            "【Long-term Interests】List 2–4 concise points describing stable, overarching themes (e.g., genres, aesthetics, recurring topics).\n"
            "【Short-term Interests】List 2–4 concise points describing currently active or emerging focuses in this batch.\n"
            "【Preference Dynamics】List 1–3 concise points describing how the user’s interests evolve (stability, novelty seeking, shifts, or diversification).\n"
            "Each bullet should be a short, declarative sentence (no more than 25 words). Avoid vague language like 'diverse interests'."
        )
    else:
        prompt += (
            f"\nThe user's previous preference profile was as follows:\n{last_preference}\n\n"
            "Now, based on both the prior profile and this new batch of videos, update and refine the user's preference summary.\n"
            "Output must follow the same structured, numbered format:\n"
            "【Long-term Interests】Update or revise previous stable themes. Add new consistent evidence if found.\n"
            "【Short-term Interests】Summarize new or temporary interests emerging from this batch.\n"
            "【Preference Dynamics】Briefly indicate whether the user’s overall pattern is stable, exploratory, or changing.\n"
            "Keep each bullet under 25 words, directly factual, and analytical. Do not output prose paragraphs."
        )

    return prompt

# 第三种 增加了 score
# def build_prompt(user, item_chunk, last_preference):
#     prompt = f"[INST] Below is a chronological list of videos recently watched by User {user}:\n"
#     for i, item in enumerate(item_chunk):
#         item = item.replace(',', '')
#         title = title_df.loc[item, 'title']
#         visual_desc = visual_df.loc[item, 'summary']
#         prompt += f"{i + 1}. {item}: Title - '{title}', Video cover description - {visual_desc}.\n"
#
#     if last_preference is None:
#         # 初始情况：没有历史长期兴趣，implicit score=0
#         prompt += (
#             "\nPlease analyze this sequence of videos to summarize the user's preferences.\n"
#             "You should describe:\n"
#             "1. **Short-term interests** (inferred from this batch only).\n"
#             "2. **Long-term interests** (initialize them based on current short-term interests).\n"
#             "3. **Implicit long-term score**: start from 0 as this is the first observation.\n"
#             "4. **Preference dynamics**: include both aspects:\n"
#             "   - Broad vs. narrow focus (does the user show diverse vs. concentrated preferences?).\n"
#             "   - Stability vs. shifts (since this is the first observation, mark as 'stable').\n\n"
#             "Write your answer in one coherent paragraph starting with:\n"
#             "'The user appears to have a preference for...'\n"
#             "At the end of your paragraph, explicitly state the implicit long-term score as a numeric value in the form: "
#             "'[Implicit Score: x.xx]'."
#         )
#     else:
#         # 后续情况：已有历史长期兴趣 + 上一个 implicit score
#         prompt += (
#             f"\nWe also know the following about the user's previous long-term preferences and implicit score:\n"
#             f"{last_preference}\n\n"
#             "Based on both the above historical knowledge and this new batch of videos, "
#             "summarize the user's preferences with a focus on:\n"
#             "1. **Short-term interests**: regenerate them from this batch of videos.\n"
#             "2. **Long-term interests**: update them by comparing the new short-term interests with the previous long-term profile. "
#             "If they are consistent, reinforce stability; if they differ, adjust the long-term profile to reflect the change.\n"
#             "3. **Implicit long-term score**: update the previous score as follows:\n"
#             "   - If the short-term interests and long-term interests are largely consistent, decrease the score.\n"
#             "   - If they differ significantly, increase the score.\n"
#             "   - If the score is ≤ 0.5, adjust within ±0.0–0.2.\n"
#             "   - If the score is > 0.5, adjust within ±0.0–0.1.\n"
#             "4. **Preference dynamics**: include both aspects:\n"
#             "   - Broad vs. narrow focus (does the user show diverse vs. concentrated preferences in this batch compared with history?).\n"
#             "   - Stability vs. shifts (decide using the implicit long-term score: "
#             "score < 0.4 → 'stable'; score ≥ 0.4 → 'shift').\n\n"
#             "Write in one continuous paragraph, concise and analytical, starting with:\n"
#             "'The user appears to have a preference for...'\n"
#             "At the end of your paragraph, explicitly state the updated implicit long-term score as a numeric value in the form: "
#             "'[Implicit Score: x.xx]'."
#         )
#
#     return prompt

# def infer(user, items, rank):
#     last_preference = None
#     print('user',user)
#     for i in range(0, len(items), 10):
#         prompt = build_prompt(user, items[i:i+10], last_preference)
#
#         messages = [
#             {"role": "user", "content": prompt},
#         ]
#
#         pipeline = pipelines[rank]
#
#         prompt = pipeline.tokenizer.apply_chat_template(
#                 messages,
#                 tokenize=False,
#                 add_generation_prompt=True
#         )
#         outputs = pipeline(
#             prompt,
#             max_new_tokens=1024,
#             eos_token_id=terminators,
#             do_sample=True,
#             temperature=0.6,
#             top_p=0.9,
#         )
#         last_preference = outputs[0]["generated_text"][len(prompt):]
#         print('preference {}'.format(i),last_preference)
#     return last_preference

def infer(user, items, rank):
    last_preference = None
    # print('user',user)
    for i in range(0, len(items), 20):
        prompt = build_prompt(user, items[i:i+20], last_preference)
        # print('len',len(items))

        messages = [
            {"role": "user", "content": prompt},
        ]

        pipeline = pipelines[rank]

        prompt = pipeline.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
        )
        outputs = pipeline(
            prompt,
            max_new_tokens=1024,
            eos_token_id=terminators,
            do_sample=True,
            temperature=0.6,
            top_p=0.9,
        )
        last_preference = outputs[0]["generated_text"][len(prompt):]
    # print('preference : ',last_preference)
    return last_preference

def gpu_computation(batch, rank):

    device = f"cuda:{rank % torch.cuda.device_count()}"
    # model.to(device)
    if rank not in pipelines:
        pipelines[rank] = transformers.pipeline(
            "text-generation",
            model=model_path,
            model_kwargs={"torch_dtype": torch.bfloat16},
            device=device,
        )
    user, items = batch['user'], batch['items']
    summaries = []
    for i in range(len(user)):
        summaries.append(infer(user[i], items[i], rank))
    print('user', user[0], 'summary', summaries[0])
    return {'user': user, 'summary': summaries}


if __name__ == "__main__":
    set_start_method("spawn")
    num_proc = 7


    chunk_size = 3000

    for i in range(0, len(user_hist_dataset), chunk_size):
        print(f"---------------------Processing chunk {i},{len(user_hist_dataset)},{chunk_size}-----------------------------------")
        sub_dataset = user_hist_dataset.select(range(i, min(i+chunk_size, len(user_hist_dataset)), 1))
        
        updated_dataset = sub_dataset.map(
            gpu_computation,
            batched=True,
            batch_size=BATCH_SIZE,
            with_rank=True,
            num_proc=num_proc
        )

        user_id = updated_dataset['user']
        summary = updated_dataset['summary']
        df = pd.DataFrame({'user_id': user_id, 'summary': summary})
        if i == 0:
            df.to_csv('user_preference_recurrent_numeric.csv', index=False, header=True, mode='w')
        else:
            df.to_csv('user_preference_recurrent_numeric.csv', index=False, header=False, mode='a')



