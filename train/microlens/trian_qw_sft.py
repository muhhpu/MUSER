# -*- coding: utf-8 -*-
# Qwen2.5-VL SFT 训练脚本（[ANS]/[REASON] 双重 loss + LoRA + 正确传入 image_grid_thw）
import os, re
from pathlib import Path
from typing import List

import torch
import numpy as np
from nltk.metrics.distance import edit_distance

import lightning as L
from torch.utils.data import DataLoader
from lightning.pytorch.callbacks import Callback, ModelCheckpoint, EarlyStopping
from lightning.pytorch.strategies import DeepSpeedStrategy

from peft import LoraConfig, get_peft_model
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

# ===== 环境变量（离线） =====
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

# ===== 超参 =====
MAX_LENGTH = 3072
GEN_MAX_NEW_TOKENS = 256
EPOCH = 4
LORA_R = 8
BATCH_SIZE = 1
NUM_WORKERS = 4
LR = 2e-5

# ===== 路径 =====
MODEL_PATH = Path("/home/team//MLLM-MSR-main/MLLM-MSR/train/Qwen2.5-VL-3B-Instruct")
SAVE_DIR = f"/home/team//MLLM-MSR-main/MLLM-MSR/save/Qwen2.5-VL-3B-Instruct-lora-e{EPOCH}-r{LORA_R}-太慢了"
DATA_ROOT = "/home/team//MLLM-MSR-main/MLLM-MSR/MicroLens-50k-training-recurrent-longshortpreference-reason-new"

# ===== 数据集 =====
from load_llava_dataset import LlavaDataset2
train_dataset = LlavaDataset2(DATA_ROOT, split="train")
val_dataset   = LlavaDataset2(DATA_ROOT, split="validation")

# ===== 小工具 =====
def normalize_ans(x: str) -> str:
    s = (x or "").strip().lower()
    return "yes" if s in ["yes", "y", "true", "1", "是", "对", "好"] else "no"

def find_sublist(big: List[int], small: List[int]) -> int:
    if not small or len(small) > len(big):
        return -1
    last = len(big) - len(small)
    for i in range(last + 1):
        if big[i:i + len(small)] == small:
            return i
    return -1

# ===== Processor & Model =====
processor = AutoProcessor.from_pretrained(
    MODEL_PATH,
    local_files_only=True,
    trust_remote_code=True,
    use_fast=False,  # 抑制 fast/slow 警告
)

model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    MODEL_PATH,
    torch_dtype=torch.bfloat16,
    local_files_only=True,
    trust_remote_code=True,
)

# ===== LoRA =====
def find_lora_targets(model: torch.nn.Module) -> List[str]:
    targets = []
    KEYWORDS = ["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj","fc1","fc2","w1","w2","w3"]
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear) and any(k in name for k in KEYWORDS):
            targets.append(name)
    return sorted(set(targets))

lora_config = LoraConfig(
    r=LORA_R, lora_alpha=32, lora_dropout=0.1, bias="none",
    target_modules=find_lora_targets(model),
    task_type="CAUSAL_LM",
)
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

# ===== collate（关键：返回 image_grid_thw） =====
def train_collate_fn(examples):
    images, texts = [], []
    for image, prompt_text, ground_truth, truth_reason in examples:
        ans = normalize_ans(ground_truth)
        user_text = (
            f"{prompt_text}\n"
            "Please answer in the following format strictly:\n"
            "[ANS] yes/no\n"
            "[REASON] a short explanation."
        )
        messages = [
            {"role": "user", "content": [{"type": "image"}, {"type": "text", "text": user_text}]}
        ]
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
        texts.append(text)
        images.append(image)

    batch = processor(
        text=texts,
        images=images,
        padding=True,
        truncation=True,
        max_length=MAX_LENGTH,
        return_tensors="pt",
    )
    # <<<<<<<<<<<< 关键：取出 image_grid_thw >>>>>>>>>>>>>
    image_grid_thw = batch.get("image_grid_thw", None)
    if image_grid_thw is None:
        # 部分版本命名不同，兜底一下
        image_grid_thw = batch.get("image_grid_thw_list", None)
    if image_grid_thw is None:
        # 再兜底：让模型至少拿到一个 list（避免 None）
        # 注意：正常情况下 processor 一定会给；如果到这里还是 None，多半是依赖版本问题
        image_grid_thw = [(1, 1, 1)] * batch["pixel_values"].shape[0]

    labels = batch["input_ids"].clone()
    labels[labels == processor.tokenizer.pad_token_id] = -100
    batch["labels"] = labels

    return (batch["input_ids"], batch["attention_mask"], batch["pixel_values"], image_grid_thw, batch["labels"])

def eval_collate_fn(examples):
    images, texts, answers = [], [], []
    for example in examples:
        if len(example) == 4:
            image, prompt_text, ground_truth, truth_reason = example
        else:
            image, prompt_text, ground_truth = example
            truth_reason = ""
        ans = ground_truth if isinstance(ground_truth, str) else ground_truth[0]
        reason = truth_reason if isinstance(ground_truth, str) else (ground_truth[1] if len(ground_truth) > 1 else "")

        user_text = f"{prompt_text}\nPlease answer in the format: [ANS] yes/no  [REASON] text."
        messages = [
            {"role": "user", "content": [{"type": "image"}, {"type": "text", "text": user_text}]}
        ]
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        texts.append(text)
        images.append(image)
        answers.append({"answer": ans, "reason": reason})

    batch = processor(
        text=texts,
        images=images,
        padding=True,
        truncation=True,
        max_length=MAX_LENGTH,
        return_tensors="pt",
    )
    image_grid_thw = batch.get("image_grid_thw", None)
    if image_grid_thw is None:
        image_grid_thw = batch.get("image_grid_thw_list", None)
    if image_grid_thw is None:
        image_grid_thw = [(1, 1, 1)] * batch["pixel_values"].shape[0]

    return (batch["input_ids"], batch["attention_mask"], batch["pixel_values"], image_grid_thw, answers)

# ===== LightningModule =====
class QwenVLSFTModule(L.LightningModule):
    def __init__(self, cfg, processor, model):
        super().__init__()
        self.cfg = cfg
        self.processor = processor
        self.model = model
        self.alpha = torch.nn.Parameter(torch.tensor(1.0))  # [ANS]
        self.beta  = torch.nn.Parameter(torch.tensor(0.5))  # [REASON]

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.cfg.get("lr", LR))

    def training_step(self, batch, batch_idx):
        input_ids, attention_mask, pixel_values, image_grid_thw, labels = batch

        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            pixel_values=pixel_values,
            image_grid_thw=image_grid_thw,  # <<<<<< 必须传
            labels=None,
        )
        logits = outputs.logits
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()

        loss_fct = torch.nn.CrossEntropyLoss(reduction="none", ignore_index=-100)
        flat_logits = shift_logits.view(-1, shift_logits.size(-1))
        flat_labels = shift_labels.view(-1)
        token_losses = loss_fct(flat_logits, flat_labels).view(shift_labels.size())

        tok = self.processor.tokenizer
        ans_seq = tok.encode("[ANS]", add_special_tokens=False)
        rea_seq = tok.encode("[REASON]", add_special_tokens=False)

        batch_answer_mask = torch.zeros_like(shift_labels, dtype=torch.float32)
        batch_reason_mask = torch.zeros_like(shift_labels, dtype=torch.float32)

        pad = tok.pad_token_id
        labels_cpu = labels.detach().cpu().tolist()
        for b, lab in enumerate(labels_cpu):
            lab_clean = [x if x != -100 else pad for x in lab]
            ans_pos = find_sublist(lab_clean, ans_seq)
            rea_pos = find_sublist(lab_clean, rea_seq)
            if ans_pos != -1:
                end = rea_pos if rea_pos != -1 else len(lab_clean)
                start_t = ans_pos + 1
                end_t = max(start_t, min(end, len(lab_clean) - 1))
                batch_answer_mask[b, start_t-1:end_t-1] = 1.0
            if rea_pos != -1:
                start_t = rea_pos + 1
                end_t = len(lab_clean) - 1
                if end_t > start_t:
                    batch_reason_mask[b, start_t-1:end_t-1] = 1.0

        eps = 1e-8
        ans_den = (batch_answer_mask.sum(dim=1) + eps)
        rea_den = (batch_reason_mask.sum(dim=1) + eps)
        ans_loss = (token_losses * batch_answer_mask).sum(dim=1) / ans_den
        rea_loss = (token_losses * batch_reason_mask).sum(dim=1) / rea_den
        loss = self.alpha * ans_loss.mean() + self.beta * rea_loss.mean()

        self.log_dict({
            "train_loss": loss.detach(),
            "ans_loss": ans_loss.mean().detach(),
            "rea_loss": rea_loss.mean().detach(),
            "alpha": self.alpha.detach(),
            "beta": self.beta.detach(),
        }, prog_bar=True, on_step=True, on_epoch=True)
        return loss

    @torch.no_grad()
    def validation_step(self, batch, batch_idx):
        input_ids, attention_mask, pixel_values, image_grid_thw, answers = batch

        gen = self.model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            pixel_values=pixel_values,
            image_grid_thw=image_grid_thw,  # <<<<<< 必须传
            max_new_tokens=GEN_MAX_NEW_TOKENS,
        )
        decoded = self.processor.batch_decode(gen, skip_special_tokens=False)

        acc = 0
        eds = []
        for pred, gold in zip(decoded, answers):
            m = re.search(r"\[ANS\]\s*(.*?)\s*\[REASON\]\s*(.*)", pred, flags=re.S)
            pa = m.group(1).strip() if m else ""
            ga = gold["answer"]
            if normalize_ans(pa) == normalize_ans(ga):
                acc += 1
            eds.append(edit_distance(pa, str(ga)) / max(len(pa), len(str(ga)), 1))

        self.log("val_acc", acc / max(1, len(decoded)), prog_bar=True)
        self.log("val_edit", float(np.mean(eds) if eds else 1.0), prog_bar=False)

# ===== Trainer =====
cfg = {"lr": LR, "max_epochs": EPOCH}
model_module = QwenVLSFTModule(cfg, processor, model)

# callbacks = [
#     ModelCheckpoint(
#         dirpath=SAVE_DIR, filename='qwen2_5_vl_3b_lora-best',
#         monitor="val_acc", mode="max", save_top_k=1
#     ),
#     EarlyStopping(monitor="val_acc", patience=3, mode="max"),
#     Callback()
# ]
callbacks = [
    ModelCheckpoint(dirpath=SAVE_DIR, filename='qwen2_5_vl_3b_lora-last', save_last=True),
]

ds_cfg = {"zero_optimization": {"stage": 2}, "bf16": {"enabled": True}, "fp16": {"enabled": False}}

trainer = L.Trainer(
    accelerator="gpu",
    devices=1,
    strategy=DeepSpeedStrategy(config=ds_cfg),
    max_epochs=EPOCH,
    precision="bf16-mixed",
    callbacks=callbacks,
    log_every_n_steps=10,
)

trainer.fit(
    model_module,
    train_dataloaders=DataLoader(
        train_dataset, collate_fn=train_collate_fn,
        batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS
    ),
    # val_dataloaders=DataLoader(
    #     val_dataset, collate_fn=eval_collate_fn,
    #     batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS
    # ),
)
