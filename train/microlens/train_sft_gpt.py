# -*- coding: utf-8 -*-
# LLaVA 训练脚本（answer+reason 双重 loss，已修正若干细节）

MAX_LENGTH = 3072          # 上下文最大长度（保留你的设置用于tokenize）
GEN_MAX_NEW_TOKENS = 256   # 生成长度建议用一个更小的值
EPOCH = 4
LORA_R = 8
MODEL_ID = "/home/team//MLLM-MSR-main/MLLM-MSR/train/llava-v1.6-mistral-7b-hf/"

REPO_ID = "yeyuyang95/llava-v1.6-mistral-7b-hf-lora"
WANDB_PROJECT = "LLaVaNeXT"
WANDB_NAME = "llava-v1.6-mistral-7b-hf-lora"

# 微调位置
# location = "yuan"
# 跨模态+llm中高层的ffn
location = "kua"


SAVE_DIR = f"/home/team//MLLM-MSR-main/MLLM-MSR/save/LLaVA/llava-v1.6-mistral-7b-hf-lora-recurrent-user-longshort-finetunereason-{location}-e{EPOCH}-r{LORA_R}"

from transformers import AutoProcessor
from transformers import BitsAndBytesConfig, LlavaNextForConditionalGeneration
import torch
from torch.utils.data import Dataset
from typing import Any, Dict
import random
from peft import LoraConfig, prepare_model_for_kbit_training, get_peft_model
from load_llava_dataset import LlavaDataset, LlavaDataset2
from PIL import ImageOps
from lightning.pytorch.loggers import WandbLogger
import lightning as L
from torch.utils.data import DataLoader
import re
import os
from nltk.metrics.distance import edit_distance
import numpy as np
from lightning.pytorch.callbacks import Callback, ModelCheckpoint
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
import logging
from huggingface_hub import HfApi

logging.getLogger("transformers").setLevel(logging.ERROR)

processor = AutoProcessor.from_pretrained(MODEL_ID)
processor.tokenizer.padding_side = "right"

# ---- 新增特殊 token: [ANS], [REASON] ----
SPECIAL_TOKENS = ["[ANS]", "[REASON]"]
added = processor.tokenizer.add_tokens(SPECIAL_TOKENS)

USE_LORA = True
USE_QLORA = False

# 可选：只保留 'yes'/'no'
def normalize_ans(x: str) -> str:
    s = (x or "").strip().lower()
    if s in ["yes", "y", "yeah", "yep", "true", "1"]:
        return "yes"
    if s in ["no", "n", "nope", "false", "0"]:
        return "no"
    # 未知时按 "no" 处理，或抛异常视数据而定
    return "no"

## Load model
if USE_QLORA or USE_LORA:
    if USE_QLORA:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.float16
        )
        model = LlavaNextForConditionalGeneration.from_pretrained(
            MODEL_ID,
            torch_dtype=torch.float16,
            quantization_config=bnb_config,
        )
    else:
        model = LlavaNextForConditionalGeneration.from_pretrained(
            MODEL_ID,
            torch_dtype=torch.float16,
            _attn_implementation="flash_attention_2",
        )
else:
    model = LlavaNextForConditionalGeneration.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.float16,
        _attn_implementation="flash_attention_2",
    )




if location == "yuan":
    def find_all_linear_names(model):
        cls = torch.nn.Linear
        lora_module_names = set()
        multimodal_keywords = ['multi_modal_projector', 'vision_model']
        for name, module in model.named_modules():
            if any(mm_keyword in name for mm_keyword in multimodal_keywords):
                continue
            if isinstance(module, cls):
                names = name.split('.')
                lora_module_names.add(names[0] if len(names) == 1 else names[-1])
        if 'lm_head' in lora_module_names:  # needed for 16-bit
            lora_module_names.remove('lm_head')
        return list(lora_module_names)


    selected = find_all_linear_names(model)

    lora_config = LoraConfig(
        r=LORA_R,
        lora_alpha=32,
        lora_dropout=0.1,
        target_modules=find_all_linear_names(model),
        init_lora_weights="gaussian",
    )
elif location == "kua":
    def find_reasoning_lora_targets(
            model,
            start_layer: int = 16,
            end_layer: int | None = None,
            include_projector: bool = True,
    ) -> list[str]:
        """
        选择 LoRA 注入目标：
          1) multi_modal_projector 的线性层（linear_1/linear_2）
          2) language_model.layers.[start_layer..end_layer] 的 FFN:
             mlp.gate_proj / mlp.up_proj / mlp.down_proj
          3) 显式排除 attention 投影（q_proj/k_proj/v_proj/o_proj）、embed/norm/rotary/lm_head 等
        返回完整模块路径列表，供 LoraConfig.target_modules 使用（endswith 匹配）。
        """
        linear_cls = torch.nn.Linear
        targets = set()

        # 自动推断 LLM 层数（例如 32 层：0..31）
        try:
            num_layers = len(model.language_model.layers)
        except Exception:
            # 回退：从命名扫描层号
            layer_ids = []
            for name, _ in model.named_modules():
                m = re.search(r"language_model\.layers\.(\d+)\.", name)
                if m:
                    layer_ids.append(int(m.group(1)))
            num_layers = (max(layer_ids) + 1) if layer_ids else 32  # 兜底

        if end_layer is None:
            end_layer = num_layers - 1

        # 1) multi_modal_projector 线性层（通常是 linear_1 / linear_2）
        if include_projector:
            for name, module in model.named_modules():
                if "multi_modal_projector" in name and isinstance(module, linear_cls):
                    # 只保留 projector 里的线性层
                    if name.endswith("linear_1") or name.endswith("linear_2"):
                        targets.add(name)

        # 2) 语言模型中高层 FFN：gate/up/down_proj（排除注意力）
        wanted_ffn_suffixes = ("mlp.gate_proj", "mlp.up_proj", "mlp.down_proj")

        for lid in range(start_layer, end_layer + 1):
            prefix = f"model.language_model.layers.{lid}."
            for suffix in wanted_ffn_suffixes:
                full = prefix + suffix
                # 确认该模块存在且是 Linear
                try:
                    mod = dict(model.named_modules())[full]
                    if isinstance(mod, linear_cls):
                        targets.add(full)
                except KeyError:
                    # 有些实现里 mlp 命名可能略有差异，放宽一次匹配
                    for name, module in model.named_modules():
                        if name.startswith(prefix) and name.endswith(suffix) and isinstance(module, linear_cls):
                            targets.add(name)

        # 3) 保险起见：移除不该注入的常见结尾（注意力、词嵌入、norm、rotary、lm_head）
        banned_suffixes = (
            "self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.o_proj",
            "embed_tokens", "rotary_emb", "lm_head",
            "input_layernorm", "post_attention_layernorm", "layer_norm", "layernorm",
        )
        targets = {t for t in targets if not any(t.endswith(x) for x in banned_suffixes)}

        # 返回排序好的列表，便于日志复现
        return sorted(targets)



    lora_config = LoraConfig(
        r=LORA_R,
        lora_alpha=32,
        lora_dropout=0.1,
        bias="none",
        # 直接把“完整模块路径”传给 target_modules。
        # PEFT 在注入时按 endswith 匹配，因此能精确命中这些层，不会波及低层或注意力层。
        target_modules=find_reasoning_lora_targets(
            model,
            start_layer=16,  # 中高层起点（可改 12/18 等）
            end_layer=31,  # 最后一层
            include_projector=True,  # 同时调 projector 的 linear_1/linear_2
        ),
        init_lora_weights="gaussian",
        task_type="CAUSAL_LM",
    )







model = prepare_model_for_kbit_training(model)
model = get_peft_model(model, lora_config)

# 如果给 tokenizer 加了新 token，要扩展模型 embedding
if added and added > 0:
    model.resize_token_embeddings(len(processor.tokenizer))

#Create PyTorch dataset
train_dataset = LlavaDataset2(
    "/home/team//MLLM-MSR-main/MLLM-MSR/MicroLens-50k-training-recurrent-longshortpreference-reason",
    split="train", sort_json_key=False
)
val_dataset = LlavaDataset2(
    "/home/team//MLLM-MSR-main/MLLM-MSR/MicroLens-50k-training-recurrent-longshortpreference-reason",
    split="validation", sort_json_key=False
)

def resize_image(image_list):
    max_width = max(img.width for img in image_list)
    max_height = max(img.height for img in image_list)
    padded_images = []
    for img in image_list:
        if img.width == max_width and img.height == max_height:
            padded_images.append(img)
            continue
        delta_width = max_width - img.width
        delta_height = max_height - img.height
        padding = (
            delta_width // 2, delta_height // 2,
            delta_width - (delta_width // 2), delta_height - (delta_height // 2)
        )
        new_img = ImageOps.expand(img, border=padding, fill='black')
        padded_images.append(new_img)
    return padded_images

# helper: find sublist in list (returns first index or -1)
def find_sublist(big, small):
    if len(small) == 0:
        return -1
    for i in range(len(big) - len(small) + 1):
        if big[i:i+len(small)] == small:
            return i
    return -1

# train_collate_fn: includes [ANS] and [REASON] in supervised target
def train_collate_fn(examples):
    images, texts = [], []
    for example in examples:
        image, prompt_text, ground_truth, truth_reason = example
        images.append(image)
        ans_norm = normalize_ans(ground_truth)
        # 重要：闭合标签改为 [/INST]
        # 训练目标部分直接拼接 [ANS] ... [REASON] ...
        prompt = f"[INST] <image>\n{prompt_text} [/INST] [ANS] {ans_norm} [REASON] {truth_reason}"
        texts.append(prompt)

    images = resize_image(images)
    batch = processor(
        text=texts, images=images, padding=True, truncation=True,
        max_length=MAX_LENGTH, return_tensors="pt"
    )

    labels = batch["input_ids"].clone()
    # pad -> -100
    labels[labels == processor.tokenizer.pad_token_id] = -100
    batch["labels"] = labels

    return (
        batch["input_ids"],
        batch["attention_mask"],
        batch["pixel_values"],
        batch["image_sizes"],
        batch["labels"]
    )

# eval_collate_fn: prompt到 [/INST] 后**加 [ANS] 引导**
def eval_collate_fn(examples):
    images, texts, answers = [], [], []
    for example in examples:
        # 训练集是四元组，这里常见是三元组；若你的 val 也有 reason，可自行适配
        if len(example) == 4:
            image, prompt_text, ground_truth, truth_reason = example
        else:
            image, prompt_text, ground_truth = example
            truth_reason = ""
        images.append(image)

        # 统一取 gold
        if isinstance(ground_truth, str):
            answer = ground_truth
            reason = truth_reason
        elif isinstance(ground_truth, (list, tuple)) and len(ground_truth) >= 1:
            answer = ground_truth[0]
            reason = ground_truth[1] if len(ground_truth) > 1 else truth_reason
        elif isinstance(ground_truth, dict):
            answer = ground_truth.get("answer", "")
            reason = ground_truth.get("reason", truth_reason)
        else:
            answer = ""
            reason = truth_reason

        # 重要：闭合标签改为 [/INST]，并在末尾加 [ANS] 引导格式化输出
        prompt = f"[INST] <image>\n{prompt_text} [/INST] [ANS]"
        texts.append(prompt)
        answers.append({"answer": answer, "reason": reason})

    images = resize_image(images)
    batch = processor(text=texts, images=images, return_tensors="pt", padding=True)
    return (
        batch["input_ids"],
        batch["attention_mask"],
        batch["pixel_values"],
        batch["image_sizes"],
        answers
    )

# LightningModule with custom token-level loss
class LlavaModelPLModule(L.LightningModule):
    def __init__(self, config, processor, model, answer_weight=1.0, reason_weight=0.5):
        super().__init__()
        self.config = config
        self.processor = processor
        self.model = model
        self.batch_size = config.get("batch_size")
        self.answer_weight = answer_weight
        self.reason_weight = reason_weight

    def training_step(self, batch, batch_idx):
        input_ids, attention_mask, pixel_values, image_sizes, labels = batch
        self.model.train()
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            pixel_values=pixel_values,
            image_sizes=image_sizes,
            labels=None
        )
        logits = outputs.logits  # (B, seq_len, vocab)

        # token-wise cross-entropy
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        vocab_size = shift_logits.size(-1)
        loss_fct = torch.nn.CrossEntropyLoss(reduction="none", ignore_index=-100)

        flat_logits = shift_logits.view(-1, vocab_size)
        flat_labels = shift_labels.view(-1)
        flat_token_losses = loss_fct(flat_logits, flat_labels)
        token_losses = flat_token_losses.view(shift_labels.size())  # (B, L-1)

        # build masks for [ANS] content and [REASON] content
        tokenizer = self.processor.tokenizer
        ans_id = tokenizer.convert_tokens_to_ids("[ANS]")
        reason_id = tokenizer.convert_tokens_to_ids("[REASON]")

        batch_answer_mask = torch.zeros_like(shift_labels, dtype=torch.float32)
        batch_reason_mask = torch.zeros_like(shift_labels, dtype=torch.float32)

        labels_cpu = labels.cpu().tolist()
        for b_idx, lab in enumerate(labels_cpu):
            pad_id = tokenizer.pad_token_id
            lab_clean = [x if x != -100 else pad_id for x in lab]

            ans_pos = find_sublist(lab_clean, [ans_id])
            reason_pos = find_sublist(lab_clean, [reason_id])

            # [ANS] content: (ans_pos+1) .. (reason_pos-1 or last_nonpad)
            if ans_pos != -1:
                ans_content_start = ans_pos + 1
                if reason_pos != -1 and reason_pos > ans_pos:
                    ans_content_end = reason_pos
                else:
                    last_nonpad = len(lab_clean)
                    for i in range(len(lab_clean)-1, -1, -1):
                        if lab_clean[i] != pad_id:
                            last_nonpad = i + 1
                            break
                    ans_content_end = last_nonpad
                s = max(ans_content_start - 1, 0)
                e = max(ans_content_end - 1, 0)
                if s < token_losses.size(1):
                    e = min(e, token_losses.size(1))
                    batch_answer_mask[b_idx, s:e] = 1.0

            # [REASON] content: (reason_pos+1) .. last_nonpad
            if reason_pos != -1:
                r_start = reason_pos + 1
                last_nonpad = len(lab_clean)
                for i in range(len(lab_clean)-1, -1, -1):
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
        loss = self.answer_weight * answer_loss + self.reason_weight * reason_loss

        bs = input_ids.size(0)
        self.log("train_loss", loss, batch_size=bs)
        self.log("train_answer_loss", answer_loss, batch_size=bs)
        self.log("train_reason_loss", reason_loss, batch_size=bs)
        return loss

    def validation_step(self, batch, batch_idx, dataset_idx=0):
        input_ids, attention_mask, pixel_values, image_sizes, answers = batch
        self.model.eval()
        generated_ids = self.model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            pixel_values=pixel_values,
            image_sizes=image_sizes,
            max_new_tokens=GEN_MAX_NEW_TOKENS,
        )
        decoded = self.processor.batch_decode(generated_ids, skip_special_tokens=False)

        ans_scores, reason_scores = [], []
        acc_count = 0

        def norm_ed(a, b):
            a, b = a or "", b or ""
            if not a and not b:
                return 0.0
            if not a or not b:
                return 1.0
            return edit_distance(a, b) / max(len(a), len(b))

        def normalize_ans(ans: str):
            ans = (ans or "").strip().lower()
            if ans in ["yes", "true", "1"]:
                return "yes"
            if ans in ["no", "false", "0"]:
                return "no"
            return ans

        for pred_str, gold in zip(decoded, answers):
            # -------- 解析预测字符串 --------
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

            # -------- 取 gold label --------
            gold_ans = gold.get("answer", "") if isinstance(gold, dict) else (
                gold[0] if isinstance(gold, (list, tuple)) else gold
            )
            gold_reason = gold.get("reason", "") if isinstance(gold, dict) else (
                gold[1] if isinstance(gold, (list, tuple)) and len(gold) > 1 else ""
            )

            # -------- 计算主任务准确率 --------
            pred_ans_norm = normalize_ans(pred_ans)
            gold_ans_norm = normalize_ans(gold_ans)
            acc = 1.0 if gold_ans_norm and (pred_ans_norm == gold_ans_norm) else 0.0
            acc_count += acc

            # -------- 计算编辑距离 --------
            ans_scores.append(norm_ed(pred_ans, gold_ans))

            # reason 只在有 gold_reason 时才评估
            if gold_reason.strip():
                reason_scores.append(norm_ed(pred_reason, gold_reason))

        mean_ans_ed = float(np.mean(ans_scores)) if ans_scores else 0.0
        mean_reason_ed = float(np.mean(reason_scores)) if reason_scores else float("nan")
        mean_acc = acc_count / len(decoded) if decoded else 0.0

        # -------- logging --------
        self.log("val_answer_edit_distance", mean_ans_ed, sync_dist=True)
        self.log("val_answer_acc", mean_acc, sync_dist=True)
        if reason_scores:  # 只有存在 reason label 才 log
            self.log("val_reason_edit_distance", mean_reason_ed, sync_dist=True)

        return {"ans_ed": mean_ans_ed, "reason_ed": mean_reason_ed, "acc": mean_acc}

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.config.get("lr"))

    def train_dataloader(self):
        return DataLoader(train_dataset, collate_fn=train_collate_fn,
                          batch_size=self.batch_size, shuffle=True, num_workers=4)

    def val_dataloader(self):
        return DataLoader(val_dataset, collate_fn=eval_collate_fn,
                          batch_size=self.batch_size, shuffle=False, num_workers=4)

# ------------- config & callbacks ----------------
config = {
    "max_epochs": EPOCH,
    "check_val_every_n_epoch": 1,
    "gradient_clip_val": 1.0,
    "accumulate_grad_batches": 4,
    "lr": 2e-5,
    "batch_size": 1,
    "num_nodes": 1,
    "warmup_steps": 50,
    "result_path": "./LLaVA",
    "verbose": True,
}

model_module = LlavaModelPLModule(config, processor, model,
                                  answer_weight=1.0, reason_weight=0.5)

api = HfApi()

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

early_stop_callback = EarlyStopping(monitor="val_answer_edit_distance",
                                    patience=3, verbose=False, mode="min")

checkpoint_callback = ModelCheckpoint(
    dirpath='./share/LLaVA/',
    filename='llava-v1.6-mistral-7b-lora-best',
    save_top_k=1,
    verbose=True,
    monitor="val_answer_edit_distance",
    mode='min'
)

trainer = L.Trainer(
    accelerator="gpu",
    devices=3,
    strategy='deepspeed_stage_2',
    max_epochs=config.get("max_epochs"),
    accumulate_grad_batches=config.get("accumulate_grad_batches"),
    check_val_every_n_epoch=config.get("check_val_every_n_epoch"),
    gradient_clip_val=config.get("gradient_clip_val"),
    precision="16-mixed",
    log_every_n_steps=10,
    limit_val_batches=5,
    num_sanity_val_steps=0,
    callbacks=[SaveToDiskCallback(), early_stop_callback, checkpoint_callback]
)

trainer.fit(model_module)
