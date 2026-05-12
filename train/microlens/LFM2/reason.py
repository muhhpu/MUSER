# -*- coding: utf-8 -*-
# LFM2-VL 训练脚本（answer+reason 双重 loss）
#
# ⚠️ 依赖:
# pip install transformers pillow
#
# 相比 Qwen 版本的修复/改动:
# 1. 模型导入/加载改为 LFM2-VL
# 2. 图像处理改为加载 PIL Image 对象 (LFM2-VL需要)
# 3. Collate Fn 中移除了 Qwen 特有的 process_vision_info 和 image_grid_thw
# 4. Chat Template 适配 LFM2-VL 的 ChatML 格式，同时保留 [ANS], [REASON] 用于 Loss 计算
# 5. 修正: 提取并传递 image_sizes 参数给 LFM2-VL 模型，解决 'NoneType' object has no attribute 'shape' 错误。

import torch
from torch.utils.data import Dataset
from typing import Any, Dict
import random
import re
import os
import numpy as np
import math
import logging
from PIL import Image
from huggingface_hub import HfApi

# ------------------ LFM2-VL 特有导入 -------------------
from transformers import AutoProcessor, AutoTokenizer, AutoModelForImageTextToText
# --------------------------------------------------

from transformers import BitsAndBytesConfig
from torch.utils.data import Dataset
from peft import LoraConfig, prepare_model_for_kbit_training, get_peft_model
# 导入你提供的 load_llava_dataset.py 里的 LlavaDataset2
from load_llava_dataset import LlavaDataset2
import lightning as L
from lightning.pytorch.loggers import WandbLogger
from torch.utils.data import DataLoader
from nltk.metrics.distance import edit_distance
from lightning.pytorch.callbacks import Callback, ModelCheckpoint
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from lightning.pytorch.strategies import DeepSpeedStrategy
from transformers.image_utils import load_image

# ------------------ 常量配置 (已更新) -------------------
MAX_LENGTH = 3072
GEN_MAX_NEW_TOKENS = 256
EPOCH = 4
LORA_R = 8

MODEL_ID = "/home/team//MLLM-MSR-main/MLLM-MSR/train/LFM2-VL-450M"

REPO_ID = "yeyuyang95/lfm2-vl-450m-lora"
WANDB_PROJECT = "LFM2-VL"
WANDB_NAME = "lfm2-vl-450m-lora-instruct-lora"

location = "kua"

SAVE_DIR = f"/home/team//MLLM-MSR-main/MLLM-MSR/save/LFM2-VL/lfm2-vl-450m-lora-recurrent-user-longshort-finetunereason-{location}-linear1-e{EPOCH}-r{LORA_R}-numeric"

logging.getLogger("transformers").setLevel(logging.ERROR)

processor = AutoProcessor.from_pretrained(MODEL_ID)
processor.tokenizer.padding_side = "right"

SPECIAL_TOKENS = ["[ANS]", "[REASON]"]
# ⚠️ LFM2-VL 的 tokenizer 可能不支持或不需要添加这些特殊 Token，但为了 Loss 计算需要其 ID
# added = processor.tokenizer.add_tokens(SPECIAL_TOKENS)

USE_LORA = True
USE_QLORA = False


def normalize_ans(x: str) -> str:
    s = (x or "").strip().lower()
    if s in ["yes", "y", "yeah", "yep", "true", "1"]:
        return "yes"
    if s in ["no", "n", "nope", "false", "0"]:
        return "no"
    return "no"


## Load model
# ⚠️ LFM2-VL 使用 AutoModelForImageTextToText
if USE_QLORA:
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.float16
    )
    model = AutoModelForImageTextToText.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.float16,
        quantization_config=bnb_config,
        trust_remote_code=True,
    )
else:
    model = AutoModelForImageTextToText.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    )

# ------------------ LoRA 配置 (已适配 LFM2-VL) -------------------
if location == "kua":
    # 针对 LFM2-VL 的 LFM2 语言模型 backbone
    lora_config = LoraConfig(
        r=LORA_R,
        lora_alpha=32,
        lora_dropout=0.1,
        bias="none",
        # 针对 LFM2-VL，使用其默认的 q_proj/v_proj 等
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        init_lora_weights="gaussian",
        task_type="CAUSAL_LM",
    )

model = prepare_model_for_kbit_training(model)
model = get_peft_model(model, lora_config)

# ⚠️ LFM2-VL的 tokenizer 可能不支持 add_tokens，为避免错误，暂时不执行 resize_token_embeddings
# if added and added > 0:
#     model.resize_token_embeddings(len(processor.tokenizer))

# ------------------ Dataset -------------------
dataset_path = "/home/team//MLLM-MSR-main/MLLM-MSR/MicroLens-50k-training-recurrent-longshortpreference-reason-new-numeric-imagePath"
train_dataset = LlavaDataset2(
    dataset_path,
    split="train", sort_json_key=False
)
val_dataset = LlavaDataset2(
    dataset_path,
    split="validation", sort_json_key=False
)


def find_sublist(big, small):
    if len(small) == 0:
        return -1
    for i in range(len(big) - len(small) + 1):
        if big[i:i + len(small)] == small:
            return i
    return -1


# ------------------ Collate Fns (已修复 image_sizes 缺失) -------------------

def train_collate_fn(examples):
    all_messages = []
    all_images = []

    for example in examples:
        # image 此时是路径
        image_path, prompt_text, ground_truth, truth_reason = example
        # ⚠️ LFM2-VL 需要 PIL Image 对象
        image = load_image(image_path)
        all_images.append(image)

        ans_norm = normalize_ans(ground_truth)

        # ⚠️ LFM2-VL Chat Template: 需要 <image> 占位符
        messages = [
            {
                "role": "system",
                "content": "You are a helpful multimodal assistant by Liquid AI."
            },
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": prompt_text}
                ]
            },
            {
                "role": "assistant",
                "content": f"[ANS] {ans_norm} [REASON] {truth_reason}"
            }
        ]
        all_messages.append(messages)

    texts = [
        processor.apply_chat_template(msg, tokenize=False, add_generation_prompt=False)
        for msg in all_messages
    ]

    batch = processor(
        text=texts,
        images=all_images,
        padding=True,
        truncation=True,
        max_length=MAX_LENGTH,
        return_tensors="pt"
    )

    labels = batch["input_ids"].clone()
    labels[labels == processor.tokenizer.pad_token_id] = -100
    batch["labels"] = labels
    spatial_shapes = batch.pop("spatial_shapes", None)
    pixel_attention_mask = batch.pop("pixel_attention_mask", None)


    # 修复: 移除 image_grid_thw，LFM2-VL不需要
    return (batch["input_ids"], batch["attention_mask"], batch["pixel_values"], spatial_shapes, batch["labels"],pixel_attention_mask)


def eval_collate_fn(examples):
    all_messages = []
    answers = []
    all_images = []

    for example in examples:
        # image 此时是路径
        image_path, prompt_text, ground_truth, truth_reason = example
        # ⚠️ LFM2-VL 需要 PIL Image 对象
        image = load_image(image_path)
        all_images.append(image)

        if isinstance(ground_truth, str):
            answer = ground_truth
            reason = truth_reason
        # ... (answer/reason parsing logic remains the same)
        elif isinstance(ground_truth, (list, tuple)) and len(ground_truth) >= 1:
            answer = ground_truth[0]
            reason = truth_reason
        elif isinstance(ground_truth, dict):
            answer = ground_truth.get("answer", "")
            reason = ground_truth.get("reason", truth_reason)
        else:
            answer = ""
            reason = truth_reason

        # ⚠️ LFM2-VL Chat Template
        messages = [
            {
                "role": "system",
                "content": "You are a helpful multimodal assistant by Liquid AI."
            },
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": prompt_text}
                ]
            }
        ]
        all_messages.append(messages)
        answers.append({"answer": answer, "reason": reason})

    texts = [
        processor.apply_chat_template(msg, tokenize=False, add_generation_prompt=True)
        for msg in all_messages
    ]

    batch = processor(
        text=texts,
        images=all_images,
        return_tensors="pt",
        padding=True
    )

    spatial_shapes = batch.pop("spatial_shapes", None)
    pixel_attention_mask = batch.pop("pixel_attention_mask", None)
    # 修复: 移除 image_grid_thw
    return (batch["input_ids"], batch["attention_mask"], batch["pixel_values"], spatial_shapes, answers,pixel_attention_mask)


# ------------------ LightningModule (已修复 image_sizes 传递) -------------------
class QwenVLModelPLModule(L.LightningModule):
    def __init__(self, config, processor, model):
        super().__init__()
        self.config = config
        self.processor = processor
        self.model = model
        self.batch_size = config.get("batch_size")

        self.ans_weight_start = float(config.get("ans_weight_start", 1.0))
        self.ans_weight_end = float(config.get("ans_weight_end", 0.5))
        self.ans_weight_schedule = str(config.get("ans_weight_schedule", "linear")).lower()

        self.register_buffer("alpha", torch.tensor(self.ans_weight_start))
        self.register_buffer("beta", torch.tensor(1.0 - self.ans_weight_start))

        self.ans_id = self.processor.tokenizer.convert_tokens_to_ids("[ANS]")
        self.reason_id = self.processor.tokenizer.convert_tokens_to_ids("[REASON]")

        if self.ans_id == self.processor.tokenizer.unk_token_id or self.reason_id == self.processor.tokenizer.unk_token_id:
            print("⚠️ 警告: [ANS] 或 [REASON] Token ID 未找到，请确保它们已被正确添加到 LFM2-VL 的 Tokenizer!")

    def on_train_batch_start(self, batch, batch_idx):
        # ... (Loss 权重调度逻辑不变) ...
        max_steps = self.trainer.max_steps or (len(self.trainer.train_dataloader) * self.trainer.max_epochs)
        if max_steps > 0:
            progress = min(max(self.global_step / max_steps, 0.0), 1.0)
        else:
            progress = 0.0

        factor = 1.0 - progress
        alpha_val = self.ans_weight_end + (self.ans_weight_start - self.ans_weight_end) * factor
        beta_val = 1.0 - alpha_val

        self.alpha = torch.tensor(alpha_val, device=self.device)
        self.beta = torch.tensor(beta_val, device=self.device)

        self.log("alpha_weight", self.alpha.item(), prog_bar=True, on_step=True, on_epoch=False, sync_dist=True)
        self.log("beta_weight", self.beta.item(), prog_bar=True, on_step=True, on_epoch=False, sync_dist=True)

    def training_step(self, batch, batch_idx):
        # 🔥 修复: 接收 image_sizes
        input_ids, attention_mask, pixel_values, image_sizes, labels,pixel_attention_mask = batch
        self.model.train()

        # 🔥 修复: 传递 image_sizes 给模型
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            pixel_values=pixel_values,
            spatial_shapes=image_sizes,  # LFM2-VL 需要这个参数
            pixel_attention_mask=pixel_attention_mask,
            labels=None
        )
        logits = outputs.logits

        # ... (Loss 计算逻辑不变) ...
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        vocab_size = shift_logits.size(-1)
        loss_fct = torch.nn.CrossEntropyLoss(reduction="none", ignore_index=-100)

        flat_logits = shift_logits.view(-1, vocab_size)
        flat_labels = shift_labels.view(-1)
        flat_token_losses = loss_fct(flat_logits, flat_labels)
        token_losses = flat_token_losses.view(shift_labels.size())

        tokenizer = self.processor.tokenizer
        ans_id = self.ans_id
        reason_id = self.reason_id

        batch_answer_mask = torch.zeros_like(shift_labels, dtype=torch.float32)
        batch_reason_mask = torch.zeros_like(shift_labels, dtype=torch.float32)

        labels_cpu = labels.cpu().tolist()
        for b_idx, lab in enumerate(labels_cpu):
            pad_id = tokenizer.pad_token_id
            lab_clean = [x if x != -100 else pad_id for x in lab]
            ans_pos = find_sublist(lab_clean, [ans_id])
            reason_pos = find_sublist(lab_clean, [reason_id])

            if ans_pos != -1:
                ans_content_start = ans_pos + 1
                if reason_pos != -1 and reason_pos > ans_pos:
                    ans_content_end = reason_pos
                else:
                    last_nonpad = len(lab_clean)
                    for i in range(len(lab_clean) - 1, -1, -1):
                        if lab_clean[i] != pad_id:
                            last_nonpad = i + 1
                            break
                    ans_content_end = last_nonpad
                s = max(ans_content_start - 1, 0)
                e = max(ans_content_end - 1, 0)
                if s < token_losses.size(1):
                    e = min(e, token_losses.size(1))
                    batch_answer_mask[b_idx, s:e] = 1.0

            if reason_pos != -1:
                r_start = reason_pos + 1
                last_nonpad = len(lab_clean)
                for i in range(len(lab_clean) - 1, -1, -1):
                    if lab_clean[i] != pad_id:
                        last_nonpad = i + 1
                        break
                r_end = last_nonpad
                s = max(r_start - 1, 0)
                e = max(r_end - 1, 0)
                if s < token_losses.size(1):
                    e = min(e, token_losses.size(1))
                    batch_reason_mask[b_idx, s:e] = 1.0

        eps = 1e-8
        answer_counts = batch_answer_mask.sum(dim=1).clamp(min=eps)
        reason_counts = batch_reason_mask.sum(dim=1).clamp(min=eps)
        answer_loss_per_item = (token_losses * batch_answer_mask).sum(dim=1) / answer_counts
        reason_loss_per_item = (token_losses * batch_reason_mask).sum(dim=1) / reason_counts

        answer_loss = answer_loss_per_item.mean()
        reason_loss = reason_loss_per_item.mean()

        # 保持 reason loss 贡献度的增加 (通过 alpha 线性调度实现)
        loss = self.alpha * answer_loss + self.beta * reason_loss

        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        self.log("train_answer_loss", answer_loss, on_step=True, on_epoch=True, logger=True, sync_dist=True)
        self.log("train_reason_loss", reason_loss, on_step=True, on_epoch=True, logger=True, sync_dist=True)

        return loss

    def validation_step(self, batch, batch_idx, dataset_idx=0):
        # 🔥 修复: 接收 image_sizes
        input_ids, attention_mask, pixel_values, image_sizes, answers,pixel_attention_mask = batch
        self.model.eval()

        # 🔥 修复: 传递 image_sizes 给模型
        generated_ids = self.model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            pixel_values=pixel_values,
            spatial_shapes=image_sizes,  # LFM2-VL 需要这个参数
            max_new_tokens=GEN_MAX_NEW_TOKENS,
            pixel_attention_mask=pixel_attention_mask,
        )
        decoded = self.processor.batch_decode(generated_ids, skip_special_tokens=False)

        # ... (Evaluation logic remains the same) ...
        ans_scores, reason_scores = [], []
        acc_count = 0

        def norm_ed(a, b):
            a, b = a or "", b or ""
            if not a and not b: return 0.0
            if not a or not b: return 1.0
            return edit_distance(a, b) / max(len(a), len(b))

        def normalize_ans_local(ans: str):
            ans = (ans or "").strip().lower()
            if ans in ["yes", "true", "1"]: return "yes"
            if ans in ["no", "false", "0"]: return "no"
            return ans

        for pred_str, gold in zip(decoded, answers):
            # ⚠️ LFM2-VL ChatML 清理: 移除系统/用户/助手的起始和结束标记
            pred_str = re.sub(r"<\|startoftext\|>", "", pred_str, flags=re.S).strip()
            pred_str = re.sub(r"<\|im_start\|>system.*?<\|im_end\|>", "", pred_str, flags=re.S).strip()
            pred_str = re.sub(r"<\|im_start\|>user.*?<\|im_end\|>", "", pred_str, flags=re.S).strip()
            pred_str = re.sub(r"<\|im_start\|>assistant", "", pred_str, flags=re.S).strip()
            pred_str = pred_str.replace("<|im_end|>", "").strip()

            # 移除图像占位符 <image>，否则会影响后续解析
            pred_str = pred_str.replace("<image>", "").strip()

            # 保持 [ANS] [REASON] 的解析逻辑不变
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

            gold_ans = gold.get("answer", "")
            gold_reason = gold.get("reason", "")

            pred_ans_norm = normalize_ans_local(pred_ans)
            gold_ans_norm = normalize_ans_local(gold_ans)
            acc = 1.0 if gold_ans_norm and (pred_ans_norm == gold_ans_norm) else 0.0
            acc_count += acc

            ans_scores.append(norm_ed(pred_ans, gold_ans))
            if gold_reason.strip():
                reason_scores.append(norm_ed(pred_reason, gold_reason))

        mean_ans_ed = float(np.mean(ans_scores)) if ans_scores else 0.0
        mean_reason_ed = float(np.mean(reason_scores)) if reason_scores else float("nan")
        mean_acc = acc_count / len(decoded) if decoded else 0.0

        self.log("val_answer_edit_distance", mean_ans_ed, sync_dist=True, batch_size=len(decoded))
        self.log("val_answer_acc", mean_acc, sync_dist=True, batch_size=len(decoded))
        if reason_scores:
            self.log("val_reason_edit_distance", mean_reason_ed, sync_dist=True, batch_size=len(decoded))

        return {"ans_ed": mean_ans_ed, "reason_ed": mean_reason_ed, "acc": mean_acc}

    def configure_optimizers(self):
        base_lr = self.config.get("lr", 1e-5)
        # 权重 alpha/beta 由 hook 控制，不参与优化
        main_params = [p for n, p in self.named_parameters() if n not in ("alpha", "beta")]
        optimizer = torch.optim.AdamW(main_params, lr=base_lr)
        return optimizer

    def train_dataloader(self):
        return DataLoader(train_dataset, collate_fn=train_collate_fn,
                          batch_size=self.batch_size, shuffle=True, num_workers=4)

    def val_dataloader(self):
        return DataLoader(val_dataset, collate_fn=eval_collate_fn,
                          batch_size=self.batch_size, shuffle=False, num_workers=4)


# ------------------ Callback & Trainer (逻辑不变) -------------------
class SaveToDiskCallback(Callback):
    def on_train_epoch_end(self, trainer, pl_module):
        if trainer.global_rank == 0:
            print(f"Saving model to disk, epoch {trainer.current_epoch}")
            pl_module.model.save_pretrained(SAVE_DIR)
            processor.save_pretrained(SAVE_DIR)

    def on_train_end(self, trainer, pl_module):
        if trainer.global_rank == 0:
            print(f"Saving model to disk after training")
            pl_module.model.save_pretrained(SAVE_DIR)
            processor.save_pretrained(SAVE_DIR)


config = {
    "max_epochs": EPOCH,
    "check_val_every_n_epoch": 1,
    "gradient_clip_val": 1.0,
    "accumulate_grad_batches": 4,  # 会被 Trainer 的 8 覆盖
    "lr": 2e-5,
    "batch_size": 1,  # micro_batch_size
    "num_nodes": 1,
    "warmup_steps": 50,
    "result_path": "./LFM2_VL",
    "verbose": True,

    # ==== 动态 α 调度参数 ====
    "ans_weight_start": 1.0,
    "ans_weight_end": 0.5,
    "ans_weight_schedule": "linear",
}

model_module = QwenVLModelPLModule(config, processor, model)

api = HfApi()

early_stop_callback = EarlyStopping(monitor="val_answer_acc",
                                    patience=3, verbose=True, mode="max")

checkpoint_callback = ModelCheckpoint(
    dirpath='./share/LFM2-VL/',
    filename='lfm2-vl-450m-lora-best',
    save_top_k=1,
    verbose=True,
    monitor="val_answer_acc",
    mode='max'
)
deepspeed_config = {
    "zero_optimization": {
        "stage": 2,
        "contiguous_gradients": True,
        "overlap_comm": True,
        "reduce_scatter": True,
        "reduce_bucket_size": 5e7,
        "allgather_bucket_size": 5e7
    },
    "bf16": {"enabled": True},
    "fp16": {"enabled": False},
    "gradient_clipping": 1.0
}

trainer = L.Trainer(
    accelerator="gpu",
    devices=1,
    strategy=DeepSpeedStrategy(config=deepspeed_config),
    precision="bf16-mixed",
    max_epochs=config.get("max_epochs"),
    accumulate_grad_batches=8,
    check_val_every_n_epoch=config.get("check_val_every_n_epoch"),
    gradient_clip_val=config.get("gradient_clip_val"),
    log_every_n_steps=10,
    limit_val_batches=5,
    num_sanity_val_steps=0,
    callbacks=[SaveToDiskCallback(), early_stop_callback]
)

trainer.fit(model_module)