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

from transformers import T5Tokenizer, T5Config, T5ForConditionalGeneration
from datasets import load_dataset, load_from_disk
# ====== 你本地项目里的模块 ======
class LlavaDataset2(Dataset):
    """
    PyTorch Dataset for LLaVa. This class takes a HuggingFace Dataset as input.

    Each row, consists of image path(png/jpg/jpeg) and ground truth data (json/jsonl/txt).
    """

    def __init__(
        self,
        dataset_name_or_path: str,

        sort_json_key: bool = True,
    ):
        super().__init__()


        self.sort_json_key = sort_json_key

        self.dataset = load_from_disk(dataset_name_or_path)
        self.dataset_length = len(self.dataset)

        self.pt_token_sequences = []
        self.gt_token_sequences = []
        self.tr_token_sequences = []
        for sample in self.dataset:
            prompt = sample["prompt"].strip()
            ground_truth = str(sample["label"])
            truth_reason = sample["answer"].strip()
            self.gt_token_sequences.append(ground_truth)
            self.pt_token_sequences.append(prompt)
            self.tr_token_sequences.append(truth_reason)

    def __len__(self) -> int:
        return self.dataset_length

    def __getitem__(self, idx: int) -> Dict:
        """
        Returns one item of the dataset.

        Returns:
            image : the original Receipt image
            prompt_sequence : tokenized prompt sequence
            target_sequence : tokenized ground truth sequence
        """
        sample = self.dataset[idx]

        # inputs
        image = sample["image"]
        prompt_sequence = self.pt_token_sequences[idx]
        target_sequence = self.gt_token_sequences[idx]  # can be more than one, e.g., DocVQA Task 1
        reason_sequence = self.tr_token_sequences[idx]  # can be more than one, e.g., DocVQA Task 1

        return image, prompt_sequence, target_sequence, reason_sequence



from modeling_p5 import P5  # P5 架构实现
MODEL_PATH = "/home/team//MLLM-MSR-main/MLLM-MSR/save/P5-baseline/"
DATA_ROOT = "file:///home/team//MLLM-MSR-main/MLLM-MSR/MicroLens-50k-test-recurrent-longshortpreference-reason-new-numeric-testreason"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MAX_SRC_LEN = 1024
MAX_TGT_LEN = 512
def p5_collate_fn(batch: List[Dict[str, Any]], pad_token_id: int):
    input_ids_list = [x["input_ids"] for x in batch]
    whole_word_ids_list = [x["whole_word_ids"] for x in batch]
    labels_list = [x["labels"] for x in batch]

    truth_reason = [x["truth_reason"] for x in batch]

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
        "truth_reason": truth_reason,
        # 可选：保留原始文本
        "source_text": [x["source_text"] for x in batch],
        "target_text": [x["target_text"] for x in batch],
    }
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
import jieba
import jieba.posseg as pseg
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge import Rouge
import numpy as np
from itertools import combinations
from tqdm import tqdm

# ===== 特征抽取：抽取 GT 与 Pred 中的名词特征 =====
def extract_features(text):
    # 清洗特殊标签
    text = text.replace("[REASON]", "").replace("[ANS]", "")

    # 分词并词性标注
    words = pseg.cut(text)

    # 只保留名词类特征
    features = [word for word, flag in words if flag.startswith("n")]

    return set(features)


# ===== 指标计算 =====
def compute_all_metrics(preds, refs):
    smooth = SmoothingFunction().method1
    rouge = Rouge()

    bleu1_scores, bleu4_scores = [], []
    r1_p, r1_r, r1_f = [], [], []
    r2_p, r2_r, r2_f = [], [], []

    unique_preds = set(preds)
    usr = len(unique_preds) / len(preds)

    pred_feature_sets = []
    gt_feature_sets = []
    global_gt_features = set()

    for pred, ref in zip(preds, refs):
        pred_f = extract_features(pred)
        gt_f = extract_features(ref)

        pred_feature_sets.append(pred_f)
        gt_feature_sets.append(gt_f)
        global_gt_features |= gt_f

        # BLEU
        bleu1_scores.append(sentence_bleu([ref.split()], pred.split(), weights=(1, 0, 0, 0), smoothing_function=smooth))
        bleu4_scores.append(sentence_bleu([ref.split()], pred.split(), weights=(0.25, 0.25, 0.25, 0.25), smoothing_function=smooth))

        # ROUGE
        scores = rouge.get_scores(pred, ref)[0]
        r1_p.append(scores['rouge-1']['p'])
        r1_r.append(scores['rouge-1']['r'])
        r1_f.append(scores['rouge-1']['f'])
        r2_p.append(scores['rouge-2']['p'])
        r2_r.append(scores['rouge-2']['r'])
        r2_f.append(scores['rouge-2']['f'])

    # ===== FMR：特征是否被匹配到 =====
    fmr = np.mean([
        1 if len(gf & pf) > 0 else 0
        for pf, gf in zip(pred_feature_sets, gt_feature_sets)
    ])

    # ===== FCR：模型生成的所有特征覆盖率 =====
    pred_global_features = set().union(*pred_feature_sets)
    fcr = len(pred_global_features) / len(global_gt_features) if len(global_gt_features) else 0

    # ===== DIV：解释之间特征重叠程度（越低越好）=====
    intersect_counts = []
    for (pf1, pf2) in combinations(pred_feature_sets, 2):
        intersect_counts.append(len(pf1 & pf2))
    div = np.mean(intersect_counts) if intersect_counts else 0

    return {
        "BLEU-1": np.mean(bleu1_scores),
        "BLEU-4": np.mean(bleu4_scores),
        "ROUGE-1_P": np.mean(r1_p),
        "ROUGE-1_R": np.mean(r1_r),
        "ROUGE-1_F": np.mean(r1_f),
        "ROUGE-2_P": np.mean(r2_p),
        "ROUGE-2_R": np.mean(r2_r),
        "ROUGE-2_F": np.mean(r2_f),
        "USR": usr,
        "FMR": fmr,
        "FCR": fcr,
        "DIV": div,
    }

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

        image, prompt_text, ground_truth, truth_reason = example


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
            "truth_reason": truth_reason,
        }
def evaluate():
    # load model
    tokenizer = T5Tokenizer.from_pretrained(MODEL_PATH)
    model = T5ForConditionalGeneration.from_pretrained(MODEL_PATH).to(DEVICE)
    model.eval()

    # only inference on validation set
    test_ds = LlavaDataset2(DATA_ROOT,  sort_json_key=False)
    val_dataset = P5MicrolensDataset(test_ds, tokenizer, split="validation")

    test_loader = DataLoader(
        val_dataset,
        batch_size=4,
        shuffle=False,
        collate_fn=lambda b: p5_collate_fn(b, tokenizer.pad_token_id),
    )

    save_file = "p5_output_results.txt"
    fw = open(save_file, "w", encoding="utf-8")



    preds, refs = [], []

    print(f"\n🚀 Start Inference... Total samples: {len(test_ds)}")

    for step, batch in enumerate(test_loader):
        input_ids = batch["input_ids"].to(DEVICE)
        attention_mask = batch["attention_mask"].to(DEVICE)
        if step % 20 == 0:
            print(f"Progress: {step}/{len(test_loader)} batches processed...")


        with torch.no_grad():
            outputs = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_length=512,
                num_beams=4
            )

        decoded = tokenizer.batch_decode(outputs, skip_special_tokens=True)
        gt_reasons = batch["truth_reason"]

        preds.extend(decoded)
        refs.extend(gt_reasons)

    # 所有样本跑完后统计指标
    metrics = compute_all_metrics(preds, refs)
    print("\n🎯 Evaluation Metrics:")
    for k, v in metrics.items():
        print(f"{k}: {v:.4f}")

    fw.close()
    print("\n🎉 Done! Results saved to:", save_file)


if __name__ == "__main__":
    evaluate()
