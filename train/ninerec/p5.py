# -*- coding: utf-8 -*-
"""
在 MicroLens 任务上复现一个 P5-style baseline：
- 架构：P5 (JointEncoder + T5 decoder)，权重初始化来自 t5-base
- 输入：你现有 LlavaDataset2 的 prompt_text（用户兴趣 + 视频信息）
- 输出：生成 "[ANS] yes/no [REASON] ..."，与现有评估逻辑完全对齐
"""

import os
import re
import math
from dataclasses import dataclass
from typing import List, Dict, Any

import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from torch import nn
from tqdm import tqdm
import numpy as np

from transformers import T5Tokenizer, T5Config

# ====== 你本地项目里的模块 ======
from load_llava_dataset import LlavaDataset2  # 直接复用你现有的 Dataset
from modeling_p5 import P5  # P5 架构实现

# ====== 一些超参数，可以按需改 ======
MODEL_NAME = "/home/team//MLLM-MSR-main/MLLM-MSR/t5-base-offline"  # 作为 P5 的 backbone
SAVE_DIR = "/home/team//MLLM-MSR-main/MLLM-MSR/save/P5-baseline-ninerec/"
os.makedirs(SAVE_DIR, exist_ok=True)

MAX_SRC_LEN = 512
MAX_TGT_LEN = 128
BATCH_SIZE = 8
EPOCHS = 3
LR = 3e-4
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"



def normalize_ans(x: str) -> str:
    s = (x or "").strip().lower()
    if s in ["yes", "y", "yeah", "yep", "true", "1"]:
        return "yes"
    if s in ["no", "n", "nope", "false", "0"]:
        return "no"
    return "no"


# ====== whole_word_ids 计算（直接搬运自 P5 的 pretrain_data） ======
def calculate_whole_word_ids(tokenized_pieces: List[str], input_ids: List[int]) -> List[int]:
    """
    P5 使用 sentencepiece 的 '▁' 前缀来划分 whole word。
    这里完全照搬 pretrain_data.py 里的逻辑。
    """
    whole_word_ids = []
    curr = 0
    for i in range(len(tokenized_pieces)):
        if tokenized_pieces[i].startswith("▁"):
            curr += 1
            whole_word_ids.append(curr)
        else:
            whole_word_ids.append(curr)
    # input_ids 最后通常是 </s>，对应一个 0
    # 这里的写法与 pretrain_data.py 保持一致
    last_item = whole_word_ids[len(input_ids) - 2]  # unused，但保持结构
    return whole_word_ids[: len(input_ids) - 1] + [0]


# ====== Wrap 你的 LlavaDataset2 => 纯文本 P5 Dataset ======
class P5MicrolensDataset(Dataset):
    def __init__(self, hf_dataset: Dataset, tokenizer: T5Tokenizer, split: str = "train"):
        super().__init__()
        self.base = hf_dataset
        self.tokenizer = tokenizer
        self.split = split

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        # LlavaDataset2: (image, prompt_text, ground_truth, truth_reason)
        example = self.base[idx]
        if len(example) == 4:
            image, prompt_text, ground_truth, truth_reason = example
        else:
            image, prompt_text, ground_truth = example
            truth_reason = ""

        ans_norm = normalize_ans(ground_truth)
        source_text = prompt_text  # 不加 [INST]/<image> 等，只保留语义部分
        target_text = f"[ANS] {ans_norm} [REASON] {truth_reason}"

        # 编码 source
        # 注意：我们要拿到 token 序列（sentencepiece piece），以计算 whole_word_ids
        tokenized_pieces = self.tokenizer.tokenize(source_text)
        input_ids = self.tokenizer.encode(
            source_text,
            add_special_tokens=True,
            max_length=MAX_SRC_LEN,
            truncation=True,
        )

        # 根据 pieces 和 input_ids 计算 whole_word_ids
        # 因为 encode 时加了 </s>，所以长度会与 pieces+1 对齐
        whole_word_ids = calculate_whole_word_ids(tokenized_pieces, input_ids)

        # 编码 target
        target_ids = self.tokenizer.encode(
            target_text,
            add_special_tokens=True,
            max_length=MAX_TGT_LEN,
            truncation=True,
        )

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "whole_word_ids": torch.tensor(whole_word_ids, dtype=torch.long),
            "labels": torch.tensor(target_ids, dtype=torch.long),
            "source_text": source_text,
            "target_text": target_text,
        }


# ====== collate_fn：padding + mask + label = -100 ======
def p5_collate_fn(batch: List[Dict[str, Any]], pad_token_id: int):
    input_ids_list = [x["input_ids"] for x in batch]
    whole_word_ids_list = [x["whole_word_ids"] for x in batch]
    labels_list = [x["labels"] for x in batch]

    # padding
    input_ids = pad_sequence(input_ids_list, batch_first=True, padding_value=pad_token_id)
    whole_word_ids = pad_sequence(whole_word_ids_list, batch_first=True, padding_value=0)
    labels = pad_sequence(labels_list, batch_first=True, padding_value=pad_token_id)

    # attention mask
    attention_mask = (input_ids != pad_token_id).long()

    # 把 label 中 pad_token 位置改成 -100，便于 ignore loss
    labels_mask = (labels != pad_token_id)
    labels[~labels_mask] = -100

    return {
        "input_ids": input_ids,
        "whole_word_ids": whole_word_ids,
        "attention_mask": attention_mask,
        "labels": labels,
        # 可选：保留原始文本
        "source_text": [x["source_text"] for x in batch],
        "target_text": [x["target_text"] for x in batch],
    }


# ====== 构建 P5 模型（以 t5-base 为 backbone） ======
def build_p5_model_and_tokenizer():
    # 本地权重路径，保持与前面 MODEL_NAME 定义一致
    tokenizer = T5Tokenizer.from_pretrained(
        MODEL_NAME,
        use_fast=False,
        local_files_only=True,
    )

    # 加上你需要的特殊 token
    special_tokens = {"additional_special_tokens": ["[ANS]", "[REASON]"]}
    tokenizer.add_special_tokens(special_tokens)

    # 加载模型配置与权重（完全离线）
    config = T5Config.from_pretrained(
        MODEL_NAME,
        local_files_only=True,
    )
    model = P5.from_pretrained(
        MODEL_NAME,
        config=config,
        local_files_only=True,
    )

    # 扩展 embedding 以适配新 Token
    model.resize_token_embeddings(len(tokenizer))

    return model, tokenizer



# ====== 训练一个简单的 P5 baseline ======
def train_p5_microlens():
    model, tokenizer = build_p5_model_and_tokenizer()
    model.to(DEVICE)
    DATA_ROOT = "file:///home/team//MLLM-MSR-main/MLLM-MSR/ninerec-training-recurrent-longshortpreference-reason-numeric-imagepath"

    # ====== 构建 Dataset / DataLoader ======
    train_base = LlavaDataset2(DATA_ROOT, split="train", sort_json_key=False)
    val_base = LlavaDataset2(DATA_ROOT, split="validation", sort_json_key=False)

    train_dataset = P5MicrolensDataset(train_base, tokenizer, split="train")
    val_dataset = P5MicrolensDataset(val_base, tokenizer, split="validation")

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=4,
        collate_fn=lambda b: p5_collate_fn(b, tokenizer.pad_token_id),
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=4,
        collate_fn=lambda b: p5_collate_fn(b, tokenizer.pad_token_id),
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)

    def eval_one_epoch():
        model.eval()
        all_ans_ed, all_reason_ed = [], []
        acc_count = 0
        total = 0

        def norm_ed(a, b):
            a, b = a or "", b or ""
            if not a and not b:
                return 0.0
            if not a or not b:
                return 1.0
            from nltk.metrics.distance import edit_distance
            return edit_distance(a, b) / max(len(a), len(b))

        def normalize_ans_eval(ans: str):
            ans = (ans or "").strip().lower()
            if ans in ["yes", "true", "1"]:
                return "yes"
            if ans in ["no", "false", "0"]:
                return "no"
            return ans

        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Eval", leave=False):
                input_ids = batch["input_ids"].to(DEVICE)
                attention_mask = batch["attention_mask"].to(DEVICE)

                # 只给 prompt，不给 label，让模型自己生成
                generated_ids = model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    max_new_tokens=MAX_TGT_LEN,
                )

                decoded = tokenizer.batch_decode(generated_ids, skip_special_tokens=False)

                # gold 需要自己从 target_text 里解析（因为我们标签是 "[ANS] ... [REASON] ..."）
                gold_texts = batch["target_text"]

                for pred_str, gold_str in zip(decoded, gold_texts):
                    # ---- 解析预测字符串中的 [ANS]/[REASON] ----
                    m = re.search(r"\[ANS\]\s*(.*?)\s*\[REASON\]\s*(.*)", pred_str, flags=re.S)
                    if m:
                        pred_ans = m.group(1).strip()
                        pred_reason = m.group(2).strip()
                    else:
                        ans_tag_idx = pred_str.find("[ANS]")
                        reason_tag_idx = pred_str.find("[REASON]")
                        if ans_tag_idx != -1:
                            if reason_tag_idx != -1 and reason_tag_idx > ans_tag_idx:
                                pred_ans = pred_str[ans_tag_idx + 5:reason_tag_idx].strip()
                                pred_reason = pred_str[reason_tag_idx + 8:].strip()
                            else:
                                pred_ans = pred_str[ans_tag_idx + 5:].strip()
                                pred_reason = ""
                        else:
                            pred_ans = ""
                            pred_reason = pred_str.strip()

                    # ---- gold 同样解析 ----
                    m2 = re.search(r"\[ANS\]\s*(.*?)\s*\[REASON\]\s*(.*)", gold_str, flags=re.S)
                    if m2:
                        gold_ans = m2.group(1).strip()
                        gold_reason = m2.group(2).strip()
                    else:
                        # 理论上不会出现，因为我们构造 target_text 时就是这样的格式
                        gold_ans, gold_reason = "", ""

                    # ---- 计算主任务准确率（yes/no） ----
                    pred_ans_norm = normalize_ans_eval(pred_ans)
                    gold_ans_norm = normalize_ans_eval(gold_ans)
                    acc = 1.0 if gold_ans_norm and (pred_ans_norm == gold_ans_norm) else 0.0
                    acc_count += acc
                    total += 1

                    # ---- 计算编辑距离 ----
                    all_ans_ed.append(norm_ed(pred_ans, gold_ans))
                    if gold_reason.strip():
                        all_reason_ed.append(norm_ed(pred_reason, gold_reason))

        mean_ans_ed = float(np.mean(all_ans_ed)) if all_ans_ed else 0.0
        mean_reason_ed = float(np.mean(all_reason_ed)) if all_reason_ed else float("nan")
        mean_acc = acc_count / total if total else 0.0
        print(f"[Eval] ans_ED={mean_ans_ed:.4f}, reason_ED={mean_reason_ed:.4f}, acc={mean_acc:.4f}")
        return mean_ans_ed, mean_reason_ed, mean_acc

    # ====== 训练循环 ======
    best_acc = 0.0
    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0.0
        step = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch}")
        for batch in pbar:
            input_ids = batch["input_ids"].to(DEVICE)
            whole_word_ids = batch["whole_word_ids"].to(DEVICE)
            attention_mask = batch["attention_mask"].to(DEVICE)
            labels = batch["labels"].to(DEVICE)

            outputs = model(
                input_ids=input_ids,
                whole_word_ids=whole_word_ids,
                attention_mask=attention_mask,
                labels=labels,
                return_dict=True,
            )
            loss = outputs.loss
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            total_loss += loss.item()
            step += 1
            pbar.set_postfix({"loss": total_loss / step})

        print(f"[Epoch {epoch}] train_loss = {total_loss / max(step,1):.4f}")

        # 每个 epoch 做一次评估
        _, _, val_acc = eval_one_epoch()
        if val_acc > best_acc:
            best_acc = val_acc
            print(f"New best acc = {best_acc:.4f}, saving model to {SAVE_DIR}")
            model.save_pretrained(SAVE_DIR)
            tokenizer.save_pretrained(SAVE_DIR)

    print(f"Training finished. Best val acc = {best_acc:.4f}")


if __name__ == "__main__":
    train_p5_microlens()
