# -*- coding: utf-8 -*-
# Fine-tune Qwen2.5-VL on yes/no classification task (LoRA + Lightning + DeepSpeed)
import os, re, logging
from pathlib import Path
import torch
import lightning as L
from torch.utils.data import DataLoader
from lightning.pytorch.callbacks import Callback, ModelCheckpoint, EarlyStopping
from lightning.pytorch.strategies import DeepSpeedStrategy
from peft import LoraConfig, get_peft_model
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
import numpy as np
from nltk import edit_distance
from PIL import ImageOps
from load_llava_dataset import LlavaDataset2

# ---------------- Environment ----------------
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
logging.getLogger("transformers").setLevel(logging.ERROR)

# ---------------- Paths & Params ----------------
EPOCH = 4
LORA_R = 8
MAX_LENGTH = 3072
LR = 2e-5
BATCH_SIZE = 1

MODEL_PATH = "/home/team//MLLM-MSR-main/MLLM-MSR/train/Qwen2.5-VL-3B-Instruct"
SAVE_DIR = f"/home/team//MLLM-MSR-main/MLLM-MSR/save/Qwen2.5-VL-3B-Instruct-Lora-YesNo-e{EPOCH}-r{LORA_R}"
DATAPATH = "/home/team//MLLM-MSR-main/MLLM-MSR/MicroLens-50k-training-recurrent-noLS-yesno"

# ---------------- Load dataset ----------------
train_dataset = LlavaDataset2(DATAPATH, split="train", sort_json_key=False)
val_dataset = LlavaDataset2(DATAPATH, split="validation", sort_json_key=False)

# ---------------- Processor & Model ----------------
processor = AutoProcessor.from_pretrained(MODEL_PATH, local_files_only=True, trust_remote_code=True, use_fast=False)
model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    MODEL_PATH, torch_dtype=torch.bfloat16, local_files_only=True, trust_remote_code=True
)

# ---------------- LoRA Config ----------------
def find_lora_targets(model: torch.nn.Module):
    targets = []
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear) and any(k in name for k in [
            "q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj","fc1","fc2","w1","w2","w3"
        ]):
            targets.append(name)
    return sorted(set(targets))

lora_config = LoraConfig(
    r=LORA_R, lora_alpha=32, lora_dropout=0.1,
    bias="none", target_modules=find_lora_targets(model),
    task_type="CAUSAL_LM"
)
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

# ---------------- Collate ----------------
def resize_image(image_list):
    max_w = max(i.width for i in image_list)
    max_h = max(i.height for i in image_list)
    out = []
    for img in image_list:
        dw, dh = max_w - img.width, max_h - img.height
        pad = (dw//2, dh//2, dw-dw//2, dh-dh//2)
        out.append(ImageOps.expand(img, border=pad, fill="black"))
    return out

def train_collate_fn(examples):
    images, texts = [], []
    for image, prompt_text, ground_truth ,_ in examples:
        ans = str(ground_truth).strip()
        messages = [
            {"role": "user", "content": [{"type": "image"}, {"type": "text", "text": f"{prompt_text}\nAnswer yes or no."}]},
            {"role": "assistant", "content": [{"type": "text", "text": ans}]},
        ]
        text = processor.apply_chat_template(messages, tokenize=False)
        texts.append(text)
        images.append(image)
    images = resize_image(images)

    batch = processor(text=texts, images=images, padding=True, truncation=True,
                      max_length=MAX_LENGTH, return_tensors="pt")
    image_grid_thw = batch.get("image_grid_thw", [(1,1,1)]*len(images))
    labels = batch["input_ids"].clone()
    labels[labels == processor.tokenizer.pad_token_id] = -100
    batch["labels"] = labels
    return batch["input_ids"], batch["attention_mask"], batch["pixel_values"], image_grid_thw, batch["labels"]

def eval_collate_fn(examples):
    images, texts, answers = [], [], []
    for image, prompt_text, ground_truth,_ in examples:
        messages = [
            {"role": "user", "content": [{"type": "image"}, {"type": "text", "text": f"{prompt_text}\nAnswer yes or no."}]}
        ]
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        texts.append(text)
        images.append(image)
        answers.append(str(ground_truth))
    images = resize_image(images)
    batch = processor(text=texts, images=images, padding=True, truncation=True,
                      max_length=MAX_LENGTH, return_tensors="pt")
    image_grid_thw = batch.get("image_grid_thw", [(1,1,1)]*len(images))
    return batch["input_ids"], batch["attention_mask"], batch["pixel_values"], image_grid_thw, answers

# ---------------- Lightning Module ----------------
class QwenVLSFTModule(L.LightningModule):
    def __init__(self, cfg, processor, model):
        super().__init__()
        self.cfg = cfg
        self.processor = processor
        self.model = model

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.cfg.get("lr", LR))

    def training_step(self, batch, batch_idx):
        input_ids, attention_mask, pixel_values, image_grid_thw, labels = batch
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask,
                             pixel_values=pixel_values, image_grid_thw=image_grid_thw, labels=labels)
        loss = outputs.loss
        self.log("train_loss", loss, batch_size=BATCH_SIZE, prog_bar=True)
        return loss

    @torch.no_grad()
    def validation_step(self, batch, batch_idx):
        input_ids, attention_mask, pixel_values, image_grid_thw, answers = batch
        gen = self.model.generate(input_ids=input_ids, attention_mask=attention_mask,
                                  pixel_values=pixel_values, image_grid_thw=image_grid_thw,
                                  max_new_tokens=50)
        preds = self.processor.batch_decode(gen, skip_special_tokens=True)
        acc = sum((p.strip().lower().startswith(a.lower())) for p,a in zip(preds,answers)) / len(answers)
        self.log("val_acc", acc, batch_size=BATCH_SIZE, sync_dist=True, prog_bar=True)
        return acc

# ---------------- Trainer ----------------
cfg = {"lr": LR, "max_epochs": EPOCH}
model_module = QwenVLSFTModule(cfg, processor, model)
callbacks = [
    ModelCheckpoint(dirpath=SAVE_DIR, filename='qwen25vl_yesno_best',
                    monitor="val_acc", mode="max", save_top_k=1),
    EarlyStopping(monitor="val_acc", patience=3, mode="max"),
    Callback()
]
ds_cfg = {"zero_optimization": {"stage": 2}, "bf16": {"enabled": True}}

trainer = L.Trainer(
    accelerator="gpu",
    devices=2,
    strategy=DeepSpeedStrategy(config=ds_cfg),
    max_epochs=EPOCH,
    precision="bf16-mixed",
    callbacks=callbacks,
    log_every_n_steps=10,
)

trainer.fit(
    model_module,
    train_dataloaders=DataLoader(train_dataset, collate_fn=train_collate_fn,
                                 batch_size=BATCH_SIZE, shuffle=True, num_workers=4),
    val_dataloaders=DataLoader(val_dataset, collate_fn=eval_collate_fn,
                               batch_size=BATCH_SIZE, shuffle=False, num_workers=4),
)
