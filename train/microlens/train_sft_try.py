# -*- coding: utf-8 -*-
# 修改的 LLaVA 训练脚本（支持 answer+reason 双重 loss）

MAX_LENGTH = 3072
EPOCH = 4
LORA_R = 8
MODEL_ID = "/home/team//MLLM-MSR-main/MLLM-MSR/train/llava-v1.6-mistral-7b-hf/"

REPO_ID = "yeyuyang95/llava-v1.6-mistral-7b-hf-lora"
WANDB_PROJECT = "LLaVaNeXT"
WANDB_NAME = "llava-v1.6-mistral-7b-hf-lora"
SAVE_DIR = f"/home/team//MLLM-MSR-main/MLLM-MSR/save/LLaVA/llava-v1.6-mistral-7b-hf-lora-recurrent-user-longshort-finetunereason-e{EPOCH}-r{LORA_R}"

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
# 修正 import
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
if added > 0:
    print(f"Added {added} special tokens to tokenizer: {SPECIAL_TOKENS}")

USE_LORA = True
USE_QLORA = False

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

# apply PEFT
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

    if 'lm_head' in lora_module_names: # needed for 16-bit
        lora_module_names.remove('lm_head')
    return list(lora_module_names)

print(find_all_linear_names(model))

lora_config = LoraConfig(
    r=LORA_R,
    lora_alpha=32,
    lora_dropout=0.1,
    target_modules=find_all_linear_names(model),
    init_lora_weights="gaussian",
)

model = prepare_model_for_kbit_training(model)
model = get_peft_model(model, lora_config)

# 如果我们给 tokenizer 加了新 token，要扩展模型 embedding
if added > 0:
    model.resize_token_embeddings(len(processor.tokenizer))
    print("Resized token embeddings to", len(processor.tokenizer))

#Create PyTorch dataset
train_dataset = LlavaDataset2("/home/team//MLLM-MSR-main/MLLM-MSR/MicroLens-50k-training-recurrent-longshortpreference-reason",  split="train", sort_json_key=False)
val_dataset = LlavaDataset2("/home/team//MLLM-MSR-main/MLLM-MSR/MicroLens-50k-training-recurrent-longshortpreference-reason", split="validation", sort_json_key=False)

def resize_image(image_list):
    max_width = max(img.width for img in image_list)
    max_height = max(img.height for img in image_list)

    padded_images = []
    for img in image_list:
        if img.width == max_width and img.height == max_height:
            padded_images.append(img)
            continue
        else:
            delta_width = max_width - img.width
            delta_height = max_height - img.height

            padding = (
                delta_width // 2, delta_height // 2, delta_width - (delta_width // 2), delta_height - (delta_height // 2))

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

# train_collate_fn: now includes [ANS] and [REASON] in ground truth
def train_collate_fn(examples):
    images = []
    texts = []
    for example in examples:
        image, prompt_text, ground_truth ,truth_reason = example

        images.append(image)


        # construct the prompt+target. keep the prompt separate from target tokens.
        # note: we include both tags in the supervised target so the model learns to generate them
        prompt = f"[INST] <image>\n{prompt_text} [\\INST] [ANS] {ground_truth} [REASON] {truth_reason}"
        texts.append(prompt)

    images = resize_image(images)

    batch = processor(text=texts, images=images, padding=True, truncation=True, max_length=MAX_LENGTH, return_tensors="pt")

    labels = batch["input_ids"].clone()
    # pad token -> -100 for loss
    labels[labels == processor.tokenizer.pad_token_id] = -100
    batch["labels"] = labels

    input_ids = batch["input_ids"]
    attention_mask = batch["attention_mask"]
    pixel_values = batch["pixel_values"]
    image_sizes = batch["image_sizes"]
    labels = batch["labels"]

    return input_ids, attention_mask, pixel_values, image_sizes, labels

# eval_collate_fn: provide prompt without target; answers list contains the ground truth for eval
def eval_collate_fn(examples):
    images = []
    texts = []
    answers = []
    for example in examples:
        image, prompt_text, ground_truth = example
        images.append(image)

        # keep same parsing as training
        if isinstance(ground_truth, str):
            answer = ground_truth
            reason = ""
        elif isinstance(ground_truth, (list, tuple)) and len(ground_truth) >= 1:
            answer = ground_truth[0]
            reason = ground_truth[1] if len(ground_truth) > 1 else ""
        elif isinstance(ground_truth, dict):
            answer = ground_truth.get("answer", "")
            reason = ground_truth.get("reason", "TESTING......")
        else:
            answer = ""
            reason = "TESTING......"

        prompt = f"[INST] <image>\n{prompt_text} [\\INST]"  # note: no target here
        texts.append(prompt)
        answers.append({"answer": answer, "reason": reason})
    images = resize_image(images)

    batch = processor(text=texts, images=images, return_tensors="pt", padding=True)

    input_ids = batch["input_ids"]
    attention_mask = batch["attention_mask"]
    pixel_values = batch["pixel_values"]
    image_sizes = batch["image_sizes"]

    return input_ids, attention_mask, pixel_values, image_sizes, answers

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

        # forward but WITHOUT labels to get logits
        outputs = self.model(input_ids=input_ids,
                             attention_mask=attention_mask,
                             pixel_values=pixel_values,
                             image_sizes=image_sizes,
                             labels=None
                             )
        logits = outputs.logits  # (B, seq_len, vocab_size)

        # compute token-wise cross entropy
        shift_logits = logits[..., :-1, :].contiguous()  # predict t+1
        shift_labels = labels[..., 1:].contiguous()      # aligned labels (with -100 for pad)

        vocab_size = shift_logits.size(-1)
        loss_fct = torch.nn.CrossEntropyLoss(reduction="none", ignore_index=-100)

        # flatten
        flat_logits = shift_logits.view(-1, vocab_size)
        flat_labels = shift_labels.view(-1)
        flat_token_losses = loss_fct(flat_logits, flat_labels)  # shape (B*(L-1),)
        token_losses = flat_token_losses.view(shift_labels.size())  # (B, L-1)

        # Build answer_mask and reason_mask per batch item by locating special token ids
        tokenizer = self.processor.tokenizer
        ans_id = tokenizer.convert_tokens_to_ids("[ANS]")
        reason_id = tokenizer.convert_tokens_to_ids("[REASON]")

        batch_answer_mask = torch.zeros_like(shift_labels, dtype=torch.float32)  # (B, L-1)
        batch_reason_mask = torch.zeros_like(shift_labels, dtype=torch.float32)

        # Note: shift_labels corresponds to labels shifted left by 1 relative to input_ids.
        # We'll search the underlying (labels) token id sequences (not shifted) to find tags,
        # then map ranges to shift_labels indices (i.e., label index t corresponds to logits index t-1).
        labels_cpu = labels.cpu().tolist()
        for b_idx, lab in enumerate(labels_cpu):
            # lab is list of token ids including pad/-100; convert -100 back to pad_id to search easily
            lab_clean = [x if x != -100 else tokenizer.pad_token_id for x in lab]
            # find positions of tag tokens in lab_clean
            ans_pos = find_sublist(lab_clean, [ans_id])
            reason_pos = find_sublist(lab_clean, [reason_id])

            # compute answer token span: tokens after [ANS] up to [REASON] or end
            if ans_pos != -1:
                start = ans_pos  # index of tag
                # answer content starts at start+1
                ans_content_start = start + 1
                if reason_pos != -1 and reason_pos > ans_pos:
                    ans_content_end = reason_pos  # up to reason tag (exclusive)
                else:
                    # until sequence end (but exclude trailing pads)
                    # find last non-pad token index
                    last_nonpad = len(lab_clean)
                    # trim trailing pad tokens
                    for i in range(len(lab_clean)-1, -1, -1):
                        if lab_clean[i] != tokenizer.pad_token_id:
                            last_nonpad = i+1
                            break
                    ans_content_end = last_nonpad
                # convert to indices in shift_labels: shift_labels corresponds to labels[...,1:]
                # so content token positions in shift_labels are indices ans_content_start-1 .. ans_content_end-2
                s = max(ans_content_start - 1, 0)
                e = max(ans_content_end - 1, 0)
                if s < token_losses.size(1):
                    e = min(e, token_losses.size(1))
                    batch_answer_mask[b_idx, s:e] = 1.0

            # reason token span
            if reason_pos != -1:
                r_start = reason_pos + 1
                # reason goes until end (or until next tag, unlikely)
                last_nonpad = len(lab_clean)
                for i in range(len(lab_clean)-1, -1, -1):
                    if lab_clean[i] != tokenizer.pad_token_id:
                        last_nonpad = i+1
                        break
                r_end = last_nonpad
                s = max(r_start - 1, 0)
                e = max(r_end - 1, 0)
                if s < token_losses.size(1):
                    e = min(e, token_losses.size(1))
                    batch_reason_mask[b_idx, s:e] = 1.0

        # compute masked losses
        # avoid zero division
        eps = 1e-8
        # sum of mask values per batch item
        answer_counts = batch_answer_mask.sum(dim=1).clamp(min=eps)
        reason_counts = batch_reason_mask.sum(dim=1).clamp(min=eps)

        # per-item losses
        answer_loss_per_item = (token_losses * batch_answer_mask).sum(dim=1) / answer_counts
        reason_loss_per_item = (token_losses * batch_reason_mask).sum(dim=1) / reason_counts

        # mean over batch
        answer_loss = answer_loss_per_item.mean()
        reason_loss = reason_loss_per_item.mean()

        loss = self.answer_weight * answer_loss + self.reason_weight * reason_loss

        batch_size = input_ids.size(0)
        self.log("train_loss", loss, batch_size=batch_size)
        self.log("train_answer_loss", answer_loss, batch_size=batch_size)
        self.log("train_reason_loss", reason_loss, batch_size=batch_size)

        return loss

    def validation_step(self, batch, batch_idx, dataset_idx=0):
        input_ids, attention_mask, pixel_values, image_sizes, answers = batch
        self.model.eval()
        # autoregressively generate token IDs
        generated_ids = self.model.generate(input_ids=input_ids, attention_mask=attention_mask,
                                       pixel_values=pixel_values, image_sizes=image_sizes, max_new_tokens=MAX_LENGTH)
        # decode entire generated string (we want to extract [ANS] and [REASON])
        decoded = self.processor.batch_decode(generated_ids, skip_special_tokens=False)
        # for predictions we want to extract after prompt; best-effort using regex
        ans_scores = []
        reason_scores = []
        acc_count = 0
        for pred_str, gold in zip(decoded, answers):
            # try to extract using regex
            # allow both "[ANS] answer [REASON] reason" or with variable spaces
            m = re.search(r"\[ANS\]\s*(.*?)\s*\[REASON\]\s*(.*)", pred_str, flags=re.S)
            if m:
                pred_ans = m.group(1).strip()
                pred_reason = m.group(2).strip()
            else:
                # fallback heuristics: try to find tags without spaces
                ans_tag_idx = pred_str.find("[ANS]")
                reason_tag_idx = pred_str.find("[REASON]")
                if ans_tag_idx != -1:
                    if reason_tag_idx != -1 and reason_tag_idx > ans_tag_idx:
                        pred_ans = pred_str[ans_tag_idx+5:reason_tag_idx].strip()
                        pred_reason = pred_str[reason_tag_idx+8:].strip()
                    else:
                        pred_ans = pred_str[ans_tag_idx+5:].strip()
                        pred_reason = ""
                else:
                    # if no tags, use everything as reason and blank answer
                    pred_ans = ""
                    pred_reason = pred_str.strip()

            gold_ans = gold.get("answer", "") if isinstance(gold, dict) else (gold[0] if isinstance(gold, (list,tuple)) else gold)
            gold_reason = gold.get("reason", "") if isinstance(gold, dict) else (gold[1] if isinstance(gold,(list,tuple)) and len(gold)>1 else "")

            # answer accuracy (exact match, case-insensitive, stripped)
            acc = 1.0 if pred_ans.strip().lower() == gold_ans.strip().lower() and len(gold_ans.strip())>0 else 0.0
            acc_count += acc

            # normalized edit distance for answer and reason (if available)
            # answer ED
            if len(pred_ans.strip())==0 and len(gold_ans.strip())==0:
                ans_ed = 0.0
            elif len(pred_ans.strip())==0 or len(gold_ans.strip())==0:
                ans_ed = 1.0
            else:
                ans_ed = edit_distance(pred_ans, gold_ans) / max(len(pred_ans), len(gold_ans))

            # reason ED (may be noisy; you can replace with ROUGE/BERTScore)
            if len(pred_reason.strip())==0 and len(gold_reason.strip())==0:
                reason_ed = 0.0
            elif len(pred_reason.strip())==0 or len(gold_reason.strip())==0:
                reason_ed = 1.0
            else:
                reason_ed = edit_distance(pred_reason, gold_reason) / max(len(pred_reason), len(gold_reason))

            ans_scores.append(ans_ed)
            reason_scores.append(reason_ed)

        # log mean normalized EDs and accuracy
        mean_ans_ed = float(np.mean(ans_scores)) if len(ans_scores)>0 else 0.0
        mean_reason_ed = float(np.mean(reason_scores)) if len(reason_scores)>0 else 0.0
        mean_acc = acc_count / len(decoded) if len(decoded)>0 else 0.0

        self.log("val_answer_edit_distance", mean_ans_ed, sync_dist=True)
        self.log("val_reason_edit_distance", mean_reason_ed, sync_dist=True)
        self.log("val_answer_acc", mean_acc, sync_dist=True)

        return {"ans_ed": mean_ans_ed, "reason_ed": mean_reason_ed, "acc": mean_acc}

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.config.get("lr"))
        return optimizer

    def train_dataloader(self):
        return DataLoader(train_dataset, collate_fn=train_collate_fn, batch_size=self.batch_size, shuffle=True, num_workers=4)

    def val_dataloader(self):
        return DataLoader(val_dataset, collate_fn=eval_collate_fn, batch_size=self.batch_size, shuffle=False, num_workers=4)

# ------------- config & callbacks  (保持你原样) ----------------
config = {"max_epochs": EPOCH,
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

model_module = LlavaModelPLModule(config, processor, model, answer_weight=1.0, reason_weight=0.5)

# callbacks (same as before)
api = HfApi()

class SaveToDiskCallback(Callback):
    def on_train_epoch_end(self, trainer, pl_module):
        if trainer.global_rank == 0:
            print(f"Saving model to disk, epoch {trainer.current_epoch}")
            pl_module.model.save_pretrained(SAVE_DIR)
            pl_module.processor.save_pretrained(SAVE_DIR)

    def on_train_end(self, trainer, pl_module):
        if trainer.global_rank == 0:
            print(f"Saving model to disk after training")
            pl_module.model.save_pretrained(SAVE_DIR)
            pl_module.processor.save_pretrained(SAVE_DIR)

early_stop_callback = EarlyStopping(monitor="val_answer_edit_distance", patience=3, verbose=False, mode="min")

checkpoint_callback = ModelCheckpoint(
    dirpath='./share/LLaVA/',
    filename='llava-v1.6-mistral-7b-lora-test',
    save_top_k=1,
    verbose=True,
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
        callbacks=[SaveToDiskCallback()]
)

trainer.fit(model_module)
