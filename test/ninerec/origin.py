import torch
from multiprocess import set_start_method
from transformers import LlavaNextProcessor, LlavaNextForConditionalGeneration
from datasets import load_from_disk
from torchvision import transforms
from PIL import ImageOps
from torch.nn.functional import softmax
from torch.cuda.amp import autocast
import os
import torch.nn.functional as F
import numpy as np
from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score, \
    accuracy_score  # 保留，但已注释掉 yes/no 任务的使用
# 移除了 from peft import PeftModel, PeftConfig
from itertools import combinations
import jieba.posseg as pseg
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge import Rouge
import numpy as np
from itertools import combinations

# --- 环境设置 (保持不变) ---
os.environ['CURL_CA_BUNDLE'] = ''
os.environ["CUDA_VISIBLE_DEVICES"] = "2"

# **重要：将 prompt_template 设置为 "reason"，因为您加载的数据集是 reason 相关的**
prompt_template = "reason"
# prompt_template = "yesno" # 原始代码为 yesno，但为了匹配数据集和您的目标，修改为 reason

# --- 模型路径设置 ---
# 原始模型ID，未微调
base_model_id = "/home/team//MLLM-MSR-main/MLLM-MSR/train/llava-v1.6-mistral-7b-hf/"

# 移除了所有 peft_model_id 的定义和 PeftConfig 的加载
# config = PeftConfig.from_pretrained(peft_model_id)

# --- 模型加载 (已修改) ---
# 1. 直接加载原始模型
model = LlavaNextForConditionalGeneration.from_pretrained(base_model_id,
                                                          # cache_dir='/data1/share/.HF_cache/',
                                                          attn_implementation="flash_attention_2",
                                                          torch_dtype=torch.float16,
                                                          # quantization_config=bnb_config
                                                          # device_map="auto"
                                                          )
# 确保模型处于评估模式
model.eval()
print(f"Base model {base_model_id} loaded and set to evaluation mode.")

# --- Processor 加载 (保持不变) ---
processor = LlavaNextProcessor.from_pretrained(base_model_id, return_tensors=torch.float16)
processor.tokenizer.pad_token = processor.tokenizer.eos_token
processor.tokenizer.add_tokens(
    ["<|image|>", "<pad>"], special_tokens=True
)

# --- 数据集加载 (保持不变) ---
# dataset = load_from_disk(f"/home/team//MLLM-MSR-main/MLLM-MSR/MicroLens-50k-train-recurrent-longshortpreference-{prompt_template}-new-numeric")
dataset = load_from_disk(
    f"/home/team//MLLM-MSR-main/MLLM-MSR/ninerec-test-recurrent-longshortpreference-reason-new-numeric-testreason")
dataset = dataset.select(range(2100))
print(dataset)


def clean_text(text):
    if "[REASON]" in text:
        return text.split("[REASON]")[-1]
    return text


Yes_id, No_id = processor.tokenizer.convert_tokens_to_ids('Yes'), processor.tokenizer.convert_tokens_to_ids('No')
yes_id, no_id = processor.tokenizer.convert_tokens_to_ids('yes'), processor.tokenizer.convert_tokens_to_ids('no')
preds, refs = [], []


def gpu_computation(batch, rank):
    device = f"cuda:{(rank or 0) % torch.cuda.device_count()}"
    model.to(device)
    # yes_logits_batch, no_logits_batch = [], [] # 移除了 yes/no logits 相关的初始化

    batch_images = batch['image']

    max_width = max(img.width for img in batch_images)
    max_height = max(img.height for img in batch_images)

    padded_images = []
    for img in batch_images:
        if img.width == max_width and img.height == max_height:
            padded_images.append(img)
            continue
        else:
            delta_width = max_width - img.width
            delta_height = max_height - img.height

            padding = (
                delta_width // 2, delta_height // 2, delta_width - (delta_width // 2),
                delta_height - (delta_height // 2))

            new_img = ImageOps.expand(img, border=padding, fill='black')
            padded_images.append(new_img)

    images_after = padded_images

    # batch_size = len(batch['image']) # 未使用，可删除
    model_inputs = processor(text=batch['prompt'], images=images_after, return_tensors="pt", padding=True).to(device)

    with torch.no_grad() and autocast():
        # output_scores=False 即可，因为只关心生成文本
        outputs = model.generate(**model_inputs, max_new_tokens=350, return_dict_in_generate=True, output_scores=False)

    # 移除了 scores 提取和 yes/no logits 计算

    sequences = outputs['sequences']

    batch['output'] = processor.batch_decode(sequences, skip_special_tokens=True)
    batch['output'] = [
        o.replace("<pad>", "").replace("\n\n", "\n").strip()
        for o in processor.batch_decode(sequences, skip_special_tokens=True)
    ]

    pred_texts_batch = []
    gt_texts_batch = []

    for i in range(len(batch['output'])):
        # 移除了 .replace("[/INST]", "") 因为 `clean_text` 已经处理了大部分
        output_text = batch['output'][i]
        if "[/INST]" in output_text:
            output_text = output_text.split("[/INST]")[-1]

        pred_text = clean_text(output_text).strip()
        gt_text = clean_text(batch['answer'][i]).strip()
        pred_texts_batch.append(pred_text)
        gt_texts_batch.append(gt_text)

    # 移除了 yes_logits_batch 和 no_logits_batch 的返回
    return {
        "pred_text": pred_texts_batch,
        "gt_text": gt_texts_batch
    }


# --- 评估指标函数 (保持不变) ---

def recall_at_k(y_true, y_prob, k):
    sorted_indices = np.argsort(-y_prob, axis=1)
    sorted_labels = np.take_along_axis(y_true, sorted_indices, axis=1)
    retrieved_positives = np.sum(sorted_labels[:, :k], axis=1)
    total_positives = np.ones_like(retrieved_positives)
    recall_scores = retrieved_positives / total_positives
    return np.mean(recall_scores)


def mrr_at_k(y_true, y_prob, k):
    sorted_indices = np.argsort(-y_prob, axis=1)
    sorted_labels = np.take_along_axis(y_true, sorted_indices, axis=1)
    reciprocal_ranks = np.zeros(y_true.shape[0])

    for i, labels in enumerate(sorted_labels[:, :k]):
        first_pos = np.where(labels == 1)[0]
        if first_pos.size > 0:
            reciprocal_ranks[i] = 1 / (first_pos[0] + 1)

    return np.mean(reciprocal_ranks)


def hit_rate_at_k(y_true, y_prob, k):
    sorted_indices = np.argsort(-y_prob, axis=1)
    sorted_labels = np.take_along_axis(y_true, sorted_indices, axis=1)
    hits = (np.sum(sorted_labels[:, :k], axis=1) > 0).astype(int)
    return np.mean(hits)


def ndcg_at_k(y_true, y_prob, k):
    def dcg_at_k(scores, k):
        discounts = np.log2(np.arange(2, k + 2))
        return np.sum((2 ** scores - 1) / discounts, axis=1)

    sorted_indices = np.argsort(-y_prob, axis=1)
    sorted_scores = np.take_along_axis(y_true, sorted_indices, axis=1)[:, :k]

    ideal_sorted_scores = np.sort(y_true, axis=1)[:, ::-1][:, :k]
    idcg_scores = dcg_at_k(ideal_sorted_scores, k)

    epsilon = 1e-10
    ndcg_scores = dcg_scores / (idcg_scores + epsilon)

    return np.mean(ndcg_scores)


def extract_features(text):
    text = text.replace("[REASON]", "").replace("[ANS]", "")
    words = pseg.cut(text)
    features = [word for word, flag in words if flag.startswith("n")]
    return set(features)


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
        bleu4_scores.append(
            sentence_bleu([ref.split()], pred.split(), weights=(0.25, 0.25, 0.25, 0.25), smoothing_function=smooth))

        # ROUGE
        try:
            scores = rouge.get_scores(pred, ref)[0]
            r1_p.append(scores['rouge-1']['p'])
            r1_r.append(scores['rouge-1']['r'])
            r1_f.append(scores['rouge-1']['f'])
            r2_p.append(scores['rouge-2']['p'])
            r2_r.append(scores['rouge-2']['r'])
            r2_f.append(scores['rouge-2']['f'])
        except ValueError:
            # ROUGE 在输入为空字符串时会抛出 ValueError，此时跳过
            pass

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
    if len(pred_feature_sets) >= 2:
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


if __name__ == "__main__":
    set_start_method("spawn")
    torch.cuda.empty_cache()

    # 启用多进程/多GPU计算
    updated_dataset = dataset.map(
        gpu_computation,
        batched=True,
        batch_size=6,
        with_rank=True,
        num_proc=1  # one process per GPU
    )

    # ===== 调用指标计算 =====
    preds = updated_dataset["pred_text"]
    refs = updated_dataset["gt_text"]
    print("\n--- Model Predictions (Partial) ---")
    print(preds[:5])
    print("\n--- Ground Truths (Partial) ---")
    print(refs[:5])

    metrics = compute_all_metrics(preds, refs)

    print("\n🎯 Explanation Evaluation Metrics:")
    for k, v in metrics.items():
        print(f"{k}: {v:.4f}")

    # 以下是 yes/no 任务的指标计算，当前代码运行的是 reason 任务，因此保持注释状态
    # updated_dataset = updated_dataset.sort("user")
    # yes_logits = torch.tensor(updated_dataset['yes_logits'])
    # no_logits = torch.tensor(updated_dataset['no_logits'])
    # labels = np.array(updated_dataset['label'])
    # yes_prob = torch.stack([no_logits, yes_logits], dim=1)
    # yes_probs = F.softmax(yes_prob, dim=1)[:, 1].cpu().numpy()
    # print("AUC: ", roc_auc_score(labels, yes_probs))
    #
    # #yes_probs = F.sigmoid(yes_prob)[:, 1].cpu().numpy()
    # yes_probs = yes_probs.reshape(-1, 21)
    # labels = labels.reshape(-1, 21)
    #
    # #print(yes_probs)
    # #print(labels)
    #
    # y_preds = np.argmax(yes_probs, axis=1)
    #
    # print("Recall@3: ", recall_at_k(labels, yes_probs, 3))
    # print("Recall@5: ", recall_at_k(labels, yes_probs, 5))
    # print("Recall@10: ", recall_at_k(labels, yes_probs, 10))
    # print("HR@3: ", hit_rate_at_k(labels, yes_probs, 3))
    # print("HR@5: ", hit_rate_at_k(labels, yes_probs, 5))
    # print("HR@10: ", hit_rate_at_k(labels, yes_probs, 10))
    # print("MRR@3: ", mrr_at_k(labels, yes_probs, 3))
    # print("MRR@5: ", mrr_at_k(labels, yes_probs, 5))
    # print("MRR@10: ", mrr_at_k(labels, yes_probs, 10))
    # print("NDCG@3: ", ndcg_at_k(labels, yes_probs, 3))
    # print("NDCG@5: ", ndcg_at_k(labels, yes_probs, 5))
    # print("NDCG@10: ", ndcg_at_k(labels, yes_probs, 10))