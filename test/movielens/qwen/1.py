import torch
import os
import numpy as np
import re
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
from datasets import load_from_disk
from peft import PeftModel, PeftConfig

# ------------------ Qwen 特有导入 -------------------
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration, BitsAndBytesConfig
from qwen_vl_utils import process_vision_info

# ----------------------------------------------------

# 设置环境变量，固定使用 CUDA 设备 3 (参考 yn.py)
os.environ['CURL_CA_BUNDLE'] = ''
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
DEVICE = torch.device(
    "cuda:0" if torch.cuda.is_available() else "cpu")

# ------------------ 模型/路径配置 (参考 g.py 和 yn.py) -------------------
# 使用 g.py 的 Qwen 路径
BASE_MODEL_ID = "/home/team//MLLM-MSR-main/MLLM-MSR/train/Qwen2.5-VL-3B-Instruct/"

# PEFT_MODEL_ID = "/home/team//MLLM-MSR-main/MLLM-MSR/save/Qwen2.5-VL/movie-qwen2.5-vl-3b-lora-recurrent-user-longshort-finetunereason-kua-linear1-e4-r8-new-llm全layer-numeric"
PEFT_MODEL_ID = "/home/team//MLLM-MSR-main/MLLM-MSR/save/Qwen2.5-VL/movie-qwen2.5-vl-3b-lora-recurrent-user-longshort-finetunereason-kua-linear1-e4-r8-new-llm全layer-numeric-双惩loss"
dataset_path = "/home/team//MLLM-MSR-main/MLLM-MSR/movielens-test-recurrent-longshortpreference-reason-numeric-imagepath"

# PEFT_MODEL_ID = "/home/team//MLLM-MSR-main/MLLM-MSR/save/Qwen2.5-VL/movie-qwen2.5-vl-3b-lora-recurrent-user-noLS-yesno-e4-r8"
# dataset_path = "/home/team//MLLM-MSR-main/MLLM-MSR/movielens-test-recurrent-noLS-yesno-imagepath"

dataset = load_from_disk(dataset_path)
dataset = dataset.select(range(1200))  # 限制数据量，参考 yn.py

# 1. 加载基座模型和处理器 (使用 g.py 的 Qwen 类和配置)
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,  # <<< 关键修改: 更改为 bfloat16 避免 FP16 溢出
    trust_remote_code=True
)

# 使用 Qwen2.5-VL 模型类
base_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    BASE_MODEL_ID,
    quantization_config=bnb_config,
    device_map="auto",
    trust_remote_code=True,
)

processor = AutoProcessor.from_pretrained(BASE_MODEL_ID, trust_remote_code=True)
processor.tokenizer.padding_side = "right"

# 2. 挂载 LoRA 权重 (参考 yn.py 挂载方式)
SPECIAL_TOKENS = ["[ANS]", "[REASON]"]
added_tokens = processor.tokenizer.add_tokens(SPECIAL_TOKENS)
if added_tokens and added_tokens > 0:
    base_model.resize_token_embeddings(len(processor.tokenizer))

model = PeftModel.from_pretrained(base_model, PEFT_MODEL_ID)
model = model.merge_and_unload()
model.eval()
model.to(DEVICE)

# 3. 确定 Yes/No Token ID (使用 yn.py 的逻辑)
try:
    YES_TOKEN_ID = processor.tokenizer.encode("Yes", add_special_tokens=False)[0]
    NO_TOKEN_ID = processor.tokenizer.encode("No", add_special_tokens=False)[0]
except IndexError:
    # 尝试更长的分词，防止分词器将 'Yes' 分成多个 token
    YES_TOKEN_ID = processor.tokenizer.encode(" Yes", add_special_tokens=False)[0]
    NO_TOKEN_ID = processor.tokenizer.encode(" No", add_special_tokens=False)[0]

print(f"Token ID for 'Yes': {YES_TOKEN_ID}")
print(f"Token ID for 'No': {NO_TOKEN_ID}")


# 4. 定义推荐系统指标函数 (从 yn.py 复制)
def recall_at_k(labels, predictions, k):
    preds_sorted_idx = np.argsort(-predictions, axis=1)
    top_k_indices = preds_sorted_idx[:, :k]
    hits = np.sum(np.take_along_axis(labels, top_k_indices, axis=1), axis=1)
    total_positives = np.sum(labels, axis=1)
    recall = np.mean(hits / np.where(total_positives > 0, total_positives, 1))
    return recall


def hit_rate_at_k(labels, predictions, k):
    preds_sorted_idx = np.argsort(-predictions, axis=1)
    top_k_indices = preds_sorted_idx[:, :k]
    hits = np.sum(np.take_along_axis(labels, top_k_indices, axis=1), axis=1)
    hit_rate = np.mean(hits > 0)
    return hit_rate


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
        discounts = np.log2(np.arange(2, k + 2))
        return np.sum((2 ** scores - 1) / discounts, axis=1)

    sorted_indices = np.argsort(-y_prob, axis=1)
    sorted_scores = np.take_along_axis(y_true, sorted_indices, axis=1)[:, :k]
    dcg_scores = dcg_at_k(sorted_scores, k)

    ideal_sorted_scores = np.sort(y_true, axis=1)[:, ::-1][:, :k]
    idcg_scores = dcg_at_k(ideal_sorted_scores, k)

    epsilon = 1e-10
    ndcg_scores = dcg_scores / (idcg_scores + epsilon)
    return np.mean(ndcg_scores)


# 5. 定义 GPU 计算函数 (适配 Qwen 的输入/Prompt/输出提取)
def gpu_computation(examples):
    device = DEVICE

    all_messages = []
    images = examples['image']

    for image, prompt_text in zip(images, examples['prompt']):
        # ⚠️ Qwen 聊天格式：使用 apply_chat_template 转换成 Qwen 格式的 Prompt
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": prompt_text}
                ]
            }
        ]
        all_messages.append(messages)

    # 转换为 Qwen chat template 文本
    texts = [
        processor.apply_chat_template(msg, tokenize=False, add_generation_prompt=True)
        for msg in all_messages
    ]

    # Qwen 视觉信息处理
    image_inputs, video_inputs = process_vision_info(all_messages)

    # 1. 批处理数据
    batch = processor(
        text=texts,
        images=image_inputs,
        videos=video_inputs,
        return_tensors="pt",
        padding="longest"
    ).to(device)  # <--- 确保输入数据在正确设备上

    # 2. 模型前向传播 (Qwen 不需要额外的 spatial_shapes 等参数)
    with torch.no_grad():
        outputs = model.forward(
            input_ids=batch.input_ids,
            attention_mask=batch.attention_mask,
            pixel_values=batch.pixel_values,
            image_grid_thw=batch.get("image_grid_thw"),  # 传入 Qwen VL 必需的 grid info
        )

    # 3. 提取 Logits
    logits = outputs.logits

    # 🚨 检查点 1: 检查模型输出的 Logits 是否包含 NaN 或 Inf
    if not torch.isfinite(logits).all():
        print("---")
        print("⚠️ 🚨 Logits Checkpoint 1 🚨 ⚠️")
        print("模型输出的 Logits 包含非有限值 (NaN 或 Inf)，这通常是FP16精度溢出所致。")
        nan_mask = ~torch.isfinite(logits)
        nan_samples = nan_mask.any(dim=-1).any(dim=-1).nonzero(as_tuple=True)[0]
        if nan_samples.numel() > 0:
            print(f"   - 发现 Logits 异常的样本在当前批次中的索引: {nan_samples[:5].tolist()}")
        print("---")

        # <<< 关键调整: 将非有限值替换为极小值 -100.0，以避免数据被过滤，并确保 Softmax 结果接近 0 (No)
        logits[nan_mask] = -100.0

        # 找到最后一个非填充 token 的位置
    last_token_indices = batch.attention_mask.sum(dim=1) - 1
    last_token_indices = last_token_indices.to(device)

    # 提取最后一个 token 的 logits
    last_token_logits = torch.gather(
        logits,
        1,
        last_token_indices.view(-1, 1, 1).expand(-1, 1, logits.size(-1))
    ).squeeze(1)

    # 4. 提取 Yes/No 的 Logits
    yes_logits = last_token_logits[:, YES_TOKEN_ID].cpu().tolist()
    no_logits = last_token_logits[:, NO_TOKEN_ID].cpu().tolist()

    return {
        'yes_logits': yes_logits,
        'no_logits': no_logits,
    }


if __name__ == '__main__':
    print(dataset)

    # 7. 单卡计算 Logits (参考 yn.py 的单进程/单卡 map)
    updated_dataset = dataset.map(
        gpu_computation,
        desc="Computing logits on GPU:3",
        batched=True,
        batch_size=6,
        num_proc=1
    )

    # 8. 收集结果和计算指标 (从 yn.py 复制)
    updated_dataset = updated_dataset.sort("user")
    yes_logits = torch.tensor(updated_dataset['yes_logits'], dtype=torch.float32)
    no_logits = torch.tensor(updated_dataset['no_logits'], dtype=torch.float32)
    labels = np.array(updated_dataset['label'])

    # 🚨 检查点 2: 检查 Yes/No Logits 是否包含 NaN/Inf (由于 Checkpoint 1 替换，这里应该不会触发)
    if not (torch.isfinite(yes_logits).all() and torch.isfinite(no_logits).all()):
        print("\n---")
        print("⚠️ 🚨 Logits Checkpoint 2 🚨 ⚠️")
        print("收集到的 Yes/No Logits 包含 NaN 或 Inf。")
        # 由于 Logits 被替换，现在我们保留所有样本

        # ⚠️ 如果 Checkpoint 2 仍触发，说明替换逻辑有问题，需要进一步检查
        print("---")

    # 计算 AUC
    yes_prob = torch.stack([no_logits, yes_logits], dim=1)

    # **移除 Softmax 溢出风险的防御性代码 (保留，防止 Logits 仍过大)**
    # 裁剪 Logits 到一个安全范围，防止指数溢出
    safe_prob = torch.clamp(yes_prob, min=-50.0, max=50.0)

    yes_probs = F.softmax(safe_prob, dim=1)[:, 1].cpu().numpy()

    # 🚨 检查点 3: 检查最终概率是否为 NaN (Softmax 结果)
    if np.isnan(yes_probs).any():
        print("\n---")
        print("⚠️ 🚨 Probability Checkpoint 3 🚨 ⚠️")
        print("**错误：** Softmax 后的概率 `yes_probs` 仍然包含 NaN。")
        print("---")
        exit()

    print("--- 整体分类指标 ---")
    print("AUC: ", roc_auc_score(labels, yes_probs))

    # 重新塑形标签和概率矩阵进行 Recommender Metrics (假设每个用户有 21 个 item)
    recommendation_list_length = 12

    # 由于 Checkpoint 1 已经替换了 NaN/Inf，现在所有 2100 个样本都应该被保留
    try:
        if len(labels) % recommendation_list_length != 0:
            print(
                f"\n!! 🚨 重塑数据形状失败: 当前数据长度 ({len(labels)}) 无法被推荐列表长度 ({recommendation_list_length}) 整除。请检查 Checkpoint 1/2 是否导致样本丢失。")
            exit()

        yes_probs = yes_probs.reshape(-1, recommendation_list_length)
        labels = labels.reshape(-1, recommendation_list_length)

    except ValueError as e:
        print(f"\n!! 🚨 重塑数据形状失败: {e}")
        exit()

    print("\n--- 排序推荐指标 (@3, @5, @10) ---")

    print("Recall@3: ", recall_at_k(labels, yes_probs, 3))
    print("Recall@5: ", recall_at_k(labels, yes_probs, 5))
    print("Recall@10: ", recall_at_k(labels, yes_probs, 10))
    print("HR@3: ", hit_rate_at_k(labels, yes_probs, 3))
    print("HR@5: ", hit_rate_at_k(labels, yes_probs, 5))
    print("HR@10: ", hit_rate_at_k(labels, yes_probs, 10))

    print("MRR@3: ", mrr_at_k(labels, yes_probs, 3))
    print("MRR@5: ", mrr_at_k(labels, yes_probs, 5))
    print("MRR@10: ", mrr_at_k(labels, yes_probs, 10))
    print("NDCG@3: ", ndcg_at_k(labels, yes_probs, 3))
    print("NDCG@5: ", ndcg_at_k(labels, yes_probs, 5))
    print("NDCG@10: ", ndcg_at_k(labels, yes_probs, 10))