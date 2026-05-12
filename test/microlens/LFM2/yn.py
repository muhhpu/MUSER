import torch
# from multiprocess import set_start_method # 移除多进程设置
# ------------------ LFM2-VL 导入 -------------------
from transformers import AutoProcessor, AutoModelForImageTextToText, BitsAndBytesConfig
# ----------------------------------------------------
from datasets import load_dataset, load_from_disk
from PIL import ImageOps
from torch.nn.functional import softmax
from torch.cuda.amp import autocast
import os
import torch.nn.functional as F
import numpy as np
from sklearn.metrics import roc_auc_score
from peft import PeftModel, PeftConfig

# 设置环境变量，固定使用 CUDA 设备 3
os.environ['CURL_CA_BUNDLE'] = ''
os.environ["CUDA_VISIBLE_DEVICES"] = "3"  # <--- 关键修改：固定只使用设备 3
DEVICE = torch.device(
    "cuda:0" if torch.cuda.is_available() else "cpu")  # 这里的 cuda:0 在 CUDA_VISIBLE_DEVICES="3" 的环境下，实际上指向的是物理设备 3

# ------------------ LFM2-VL 配置 (保持不变) -------------------
BASE_MODEL_ID = "/home/team//MLLM-MSR-main/MLLM-MSR/train/LFM2-VL-450M"
# PEFT_MODEL_ID = f"/home/team//MLLM-MSR-main/MLLM-MSR/save/LFM2-VL/lfm2-vl-450m-lora-yesno_lfm2-e4-r8"
# dataset = load_from_disk(f"/home/team//MLLM-MSR-main/MLLM-MSR/MicroLens-50k-test-recurrent_noLS-yesno-imagepath")

PEFT_MODEL_ID = f"/home/team//MLLM-MSR-main/MLLM-MSR/save/LFM2-VL/lfm2-vl-450m-lora-recurrent-user-longshort-finetunereason-kua-linear1-e4-r8-numeric"
dataset = load_from_disk(f"/home/team//MLLM-MSR-main/MLLM-MSR/MicroLens-50k-test-recurrent-longshortpreference-reason-new-numeric-imagepath")
dataset = dataset.select(range(2100))

config = PeftConfig.from_pretrained(PEFT_MODEL_ID)
# ----------------------------------------------------

# 1. 加载基座模型和处理器
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    trust_remote_code=True
)

# 使用 device_map="auto" (它会根据 CUDA_VISIBLE_DEVICES 自动映射到可见设备)
base_model = AutoModelForImageTextToText.from_pretrained(
    BASE_MODEL_ID,
    quantization_config=bnb_config,
    device_map="auto",
    trust_remote_code=True,
)

processor = AutoProcessor.from_pretrained(BASE_MODEL_ID, trust_remote_code=True)
processor.tokenizer.padding_side = "right"

# 2. 挂载 LoRA 权重
SPECIAL_TOKENS = ["[ANS]", "[REASON]"]
added_tokens = processor.tokenizer.add_tokens(SPECIAL_TOKENS)
if added_tokens and added_tokens > 0:
    base_model.resize_token_embeddings(len(processor.tokenizer))

model = PeftModel.from_pretrained(base_model, PEFT_MODEL_ID)
model = model.merge_and_unload()
model.eval()
model.to(DEVICE)  # <--- 确保模型移动到 DEVICE (物理设备 3)

# 3. 确定 Yes/No Token ID
YES_TOKEN_ID = processor.tokenizer.encode("Yes", add_special_tokens=False)[0]
NO_TOKEN_ID = processor.tokenizer.encode("No", add_special_tokens=False)[0]

print(f"Token ID for 'Yes': {YES_TOKEN_ID}")
print(f"Token ID for 'No': {NO_TOKEN_ID}")


# ----------------------------------------------------

# 4. 定义 recall@k, HR@k, auc metric (保持不变)
def recall_at_k(labels, predictions, k):
    preds_sorted_idx = np.argsort(-predictions, axis=1)
    top_k_indices = preds_sorted_idx[:, :k]
    hits = np.sum(np.take_along_axis(labels, top_k_indices, axis=1), axis=1)
    # 假设每个样本至少有一个正样本
    total_positives = np.sum(labels, axis=1)
    # 处理 total_positives 为 0 的情况，避免除以零
    recall = np.mean(hits / np.where(total_positives > 0, total_positives, 1))
    return recall


def hit_rate_at_k(labels, predictions, k):
    preds_sorted_idx = np.argsort(-predictions, axis=1)
    top_k_indices = preds_sorted_idx[:, :k]
    hits = np.sum(np.take_along_axis(labels, top_k_indices, axis=1), axis=1)
    hit_rate = np.mean(hits > 0)
    return hit_rate


# --- 新增 MRR 和 NDCG ---
def mrr_at_k(y_true, y_prob, k):
    sorted_indices = np.argsort(-y_prob, axis=1)
    sorted_labels = np.take_along_axis(y_true, sorted_indices, axis=1)
    reciprocal_ranks = np.zeros(y_true.shape[0])

    for i, labels in enumerate(sorted_labels[:, :k]):
        first_pos = np.where(labels == 1)[0]
        if first_pos.size > 0:
            reciprocal_ranks[i] = 1 / (first_pos[0] + 1)

    return np.mean(reciprocal_ranks)


def ndcg_at_k(y_true, y_prob, k):
    def dcg_at_k(scores, k):
        # 计算 Discounted Cumulative Gain
        # scores 形状应为 (n_samples, k)
        discounts = np.log2(np.arange(2, k + 2))
        return np.sum((2 ** scores - 1) / discounts, axis=1)

    sorted_indices = np.argsort(-y_prob, axis=1)
    # 获取前 k 个位置的真实标签作为分数
    sorted_scores = np.take_along_axis(y_true, sorted_indices, axis=1)[:, :k]

    dcg_scores = dcg_at_k(sorted_scores, k)

    # 计算 Ideal DCG
    ideal_sorted_scores = np.sort(y_true, axis=1)[:, ::-1][:, :k]
    idcg_scores = dcg_at_k(ideal_sorted_scores, k)

    epsilon = 1e-10  # 防止除以零
    ndcg_scores = dcg_scores / (idcg_scores + epsilon)

    return np.mean(ndcg_scores)


# -------------------------

# 5. 定义 GPU 计算函数 (修改为单卡计算)
# 移除 rank 参数，直接使用预设的 DEVICE
def gpu_computation(examples):
    device = DEVICE  # <--- 使用全局定义的 DEVICE
    # model.to(device) # 模型已在主线程中移动

    texts = []
    images = [[i] for i in examples['image']]
    batch_size = len(images)

    for prompt_text in examples['prompt']:
        text = f"USER: <image>\n{prompt_text}\nASSISTANT:"
        texts.append(text)

    # 1. 批处理数据
    try:
        # 数据加载到预设的 DEVICE
        batch = processor(
            text=texts,
            images=images,
            return_tensors="pt",
            padding="longest"
        ).to(device)  # <--- 确保输入数据在正确设备上
        # 兼容 LFM2-VL 模型输入的修改
        spatial_shapes = batch.pop("spatial_shapes", None)
        pixel_attention_mask = batch.pop("pixel_attention_mask", None)

    except ValueError as e:
        print(f"Processor ValueError: {e}")
        print(f"Batch size: {batch_size}, Images provided: {len(images)}")
        print(f"First text: {texts[0]}")
        raise

    # 2. 模型前向传播
    with torch.no_grad():
        outputs = model.forward(
            input_ids=batch.input_ids,
            attention_mask=batch.attention_mask,
            pixel_values=batch.pixel_values,
            pixel_attention_mask=pixel_attention_mask,
            spatial_shapes=spatial_shapes,
        )

    # 3. 提取 Logits
    logits = outputs.logits

    last_token_indices = batch.attention_mask.sum(dim=1) - 1
    last_token_indices = last_token_indices.to(device)
    last_token_logits = torch.gather(
        logits,
        1,
        last_token_indices.view(-1, 1, 1).expand(-1, 1, logits.size(-1))
    ).squeeze(1)

    # 4. 提取 Yes/No 的 Logits
    # .cpu().tolist() 确保结果在 CPU 上
    yes_logits = last_token_logits[:, YES_TOKEN_ID].cpu().tolist()
    no_logits = last_token_logits[:, NO_TOKEN_ID].cpu().tolist()

    return {
        'yes_logits': yes_logits,
        'no_logits': no_logits,
    }


if __name__ == '__main__':
    # try:
    #     set_start_method('spawn', force=True) # <--- 移除
    # except RuntimeError:
    #     pass

    print(dataset)

    # 7. 单卡计算 Logits
    # <--- 关键修改：num_proc=1，移除 with_rank
    updated_dataset = dataset.map(
        gpu_computation,
        desc="Computing logits on GPU:3",
        batched=True,
        batch_size=6,
        num_proc=1  # <--- 关键修改：设置为单进程
        # with_rank=True # <--- 移除 rank 参数
    )

    # 8. 收集结果和计算指标
    updated_dataset = updated_dataset.sort("user")
    yes_logits = torch.tensor(updated_dataset['yes_logits'])
    no_logits = torch.tensor(updated_dataset['no_logits'])
    labels = np.array(updated_dataset['label'])

    # 计算 AUC (Logits转概率)
    yes_prob = torch.stack([no_logits, yes_logits], dim=1)
    yes_probs = F.softmax(yes_prob, dim=1)[:, 1].cpu().numpy()
    print("--- 整体分类指标 ---")
    print("AUC: ", roc_auc_score(labels, yes_probs))

    # 重新塑形标签和概率矩阵进行 Recommender Metrics
    # 假设你的数据集中每个用户的推荐列表长度为 21
    # 请根据你的实际数据集结构调整这里的维度！
    # 如果你的数据集是 MicroLens-50k-test-recurrent_noLS-yesno-imagepath, 2100条数据，
    # 且用户ID已排序，那么 2100 / 100 = 21 (假设有100个用户)

    # 检查数据总量和形状，这里保留21作为示例
    recommendation_list_length = 21  # 假设每个用户有21个item

    try:
        yes_probs = yes_probs.reshape(-1, recommendation_list_length)
        labels = labels.reshape(-1, recommendation_list_length)
    except ValueError as e:
        print(f"\n!! 🚨 重塑数据形状失败: {e}")
        print(f"!! 请检查你的数据集大小 ({len(labels)}) 是否能被推荐列表长度 ({recommendation_list_length}) 整除。")
        print("!! 跳过推荐指标计算。")
        exit()

    print("\n--- 排序推荐指标 (@3, @5, @10) ---")

    # 计算 Recall 和 HR
    print("Recall@3: ", recall_at_k(labels, yes_probs, 3))
    print("Recall@5: ", recall_at_k(labels, yes_probs, 5))
    print("Recall@10: ", recall_at_k(labels, yes_probs, 10))
    print("HR@3: ", hit_rate_at_k(labels, yes_probs, 3))
    print("HR@5: ", hit_rate_at_k(labels, yes_probs, 5))
    print("HR@10: ", hit_rate_at_k(labels, yes_probs, 10))

    # 计算 MRR 和 NDCG (新添加)
    print("MRR@3: ", mrr_at_k(labels, yes_probs, 3))
    print("MRR@5: ", mrr_at_k(labels, yes_probs, 5))
    print("MRR@10: ", mrr_at_k(labels, yes_probs, 10))
    print("NDCG@3: ", ndcg_at_k(labels, yes_probs, 3))
    print("NDCG@5: ", ndcg_at_k(labels, yes_probs, 5))
    print("NDCG@10: ", ndcg_at_k(labels, yes_probs, 10))