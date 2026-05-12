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
# from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score, accuracy_score # 未使用，可删除
from itertools import combinations
import jieba.posseg as pseg
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge import Rouge
import numpy as np
from itertools import combinations

# --- 环境设置 (保持不变) ---
os.environ['CURL_CA_BUNDLE'] = ''
os.environ["CUDA_VISIBLE_DEVICES"] = "1,3,5,6"
# 保持为 reason，因为您加载的数据集是 reason 相关的
# prompt_template = "yesno"
prompt_template = "reason"

# --- 模型路径设置 ---
# 原始模型ID，未微调
base_model_id = "/home/team//MLLM-MSR-main/MLLM-MSR/train/llava-v1.6-mistral-7b-hf/"

# --- 模型加载 (已修改) ---
# 1. 直接加载原始模型
model = LlavaNextForConditionalGeneration.from_pretrained(base_model_id,
                                                          # cache_dir='/data1/share/.HF_cache/',
                                                          attn_implementation="flash_attention_2",
                                                          torch_dtype=torch.float16,
                                                          # device_map="auto" # 在多进程/多GPU运行时通常不设置
                                                          )

# 确保模型处于评估模式
model.eval()

# --- Processor 加载 (保持不变) ---
processor = LlavaNextProcessor.from_pretrained(base_model_id, return_tensors=torch.float16)
processor.tokenizer.pad_token = processor.tokenizer.eos_token
processor.tokenizer.add_tokens(
    ["<|image|>", "<pad>"], special_tokens=True
)
print(f"Base model {base_model_id} loaded and set to evaluation mode.")

# --- 数据集加载 (保持不变) ---
dataset = load_from_disk(
    f"/home/team//MLLM-MSR-main/MLLM-MSR/movielens-test-recurrent-longshortpreference-reason-numeric-testreason")
print(dataset)


# --- 辅助函数 (保持不变) ---
def clean_text(text):
    if "[REASON]" in text:
        return text.split("[REASON]")[-1]
    return text


Yes_id, No_id = processor.tokenizer.convert_tokens_to_ids('Yes'), processor.tokenizer.convert_tokens_to_ids('No')
yes_id, no_id = processor.tokenizer.convert_tokens_to_ids('yes'), processor.tokenizer.convert_tokens_to_ids('no')
preds, refs = [], []


def gpu_computation(batch, rank):
    device = f"cuda:{(rank or 0) % torch.cuda.device_count()}"
    # 将模型移动到当前进程的GPU上
    model.to(device)

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

    model_inputs = processor(text=batch['prompt'], images=images_after, return_tensors="pt", padding=True).to(device)

    with torch.no_grad() and autocast():
        # 对于 reason 任务，我们不关心 yes/no logits，只关心生成的文本
        outputs = model.generate(**model_inputs, max_new_tokens=350, return_dict_in_generate=True,
                                 output_scores=False)  # 移除了 output_scores=True 以简化

    sequences = outputs['sequences']

    batch['output'] = processor.batch_decode(sequences, skip_special_tokens=True)
    batch['output'] = [
        o.replace("<pad>", "").replace("\n\n", "\n").strip()
        for o in processor.batch_decode(sequences, skip_special_tokens=True)
    ]

    pred_texts_batch = []
    gt_texts_batch = []

    for i in range(len(batch['output'])):
        # 移除模型输入提示符
        output_text = batch['output'][i]
        if "[/INST]" in output_text:
            output_text = output_text.split("[/INST]")[-1]

        pred_text = clean_text(output_text).strip()
        gt_text = clean_text(batch['answer'][i]).strip()
        pred_texts_batch.append(pred_text)
        gt_texts_batch.append(gt_text)

    return {
        "pred_text": pred_texts_batch,
        "gt_text": gt_texts_batch
    }


# 以下评估指标函数用于计算 reason 任务的性能 (保持不变)
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
        # ... (与原文件一致，省略以节省篇幅，但请在实际运行中保留)
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
    # 确保组合至少有两个元素
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
        num_proc=4  # one process per GPU
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

    # 注释掉 yesno 任务的指标计算，因为您当前跑的是 reason 任务
    # # updated_dataset = updated_dataset.sort("user")
    # # yes_logits = torch.tensor(updated_dataset['yes_logits'])
    # # no_logits = torch.tensor(updated_dataset['no_logits'])
    # # ... (AUC, Recall@K, HR@K, MRR@K, NDCG@K 的计算代码已注释或删除)