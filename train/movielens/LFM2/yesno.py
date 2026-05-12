import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoProcessor, BitsAndBytesConfig, AutoModelForImageTextToText # <--- 修复: 替换为 AutoModelForImageTextToText
from peft import LoraConfig, prepare_model_for_kbit_training, get_peft_model
from lightning.pytorch.loggers import WandbLogger
import lightning as L
from lightning.pytorch.callbacks import Callback, ModelCheckpoint, EarlyStopping
from load_llava_dataset import LlavaDataset2  # 假设这是你的数据集文件
from nltk import edit_distance
import numpy as np
import logging
import re
import os
from huggingface_hub import HfApi

# ------------------ LFM2-VL-450M 配置 (已修改) -------------------
MAX_LENGTH = 3072
EPOCH = 4
LORA_R = 8

# ⚠️ 目标模型路径改为 LFM2-VL-450M
MODEL_ID = "/home/team//MLLM-MSR-main/MLLM-MSR/train/LFM2-VL-450M"

prompt_template = "yesno_lfm2"  # 更改名称以区分
REPO_ID = "/lfm2-vl-450m-lora"
WANDB_PROJECT = "LFM2-VL"
WANDB_NAME = f"lfm2-vl-450m-lora-{prompt_template}"
SAVE_DIR = f"/home/team//MLLM-MSR-main/MLLM-MSR/save/LFM2-VL/movie-lfm2-vl-450m-lora-{prompt_template}-e{EPOCH}-r{LORA_R}"

# 保持数据集路径不变
datapath = f"movielens-training-recurrent-longshortpreference-reason-numeric-imagepath"
logging.getLogger("transformers").setLevel(logging.ERROR)

# ------------------ 加载 Processor 和 Model -------------------

# ⚠️ 使用 AutoProcessor
processor = AutoProcessor.from_pretrained(MODEL_ID, trust_remote_code=True)
processor.tokenizer.padding_side = "right"

# ⚠️ 修复：LFM2/LLaVA 等模型通常需要这些特殊 Token，保持不变
SPECIAL_TOKENS = ["[ANS]", "[REASON]"]
added = processor.tokenizer.add_tokens(SPECIAL_TOKENS)

USE_LORA = True
USE_QLORA = False

## Load model
# 🔥 修复：使用 AutoModelForImageTextToText
LFM2_MODEL_CLASS = AutoModelForImageTextToText
if USE_QLORA or USE_LORA:
    if USE_QLORA:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.float16
        )
        model = LFM2_MODEL_CLASS.from_pretrained(
            MODEL_ID,
            torch_dtype=torch.float16,
            quantization_config=bnb_config,
            trust_remote_code=True,
        )
    else:
        model = LFM2_MODEL_CLASS.from_pretrained(
            MODEL_ID,
            torch_dtype=torch.float16,
            # LFM2 模型通常使用标准 attention，移除 Qwen 的 _attn_implementation
            trust_remote_code=True,
        )
else:
    model = LFM2_MODEL_CLASS.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.float16,
        trust_remote_code=True,
    )

# ⚠️ 修复：如果添加了 token，需要 resize
if added and added > 0:
    model.resize_token_embeddings(len(processor.tokenizer))


# Apply PEFT
# ⚠️ LoRA Target Modules 适配 LLaMA/Transformer 架构
def find_lfm2_lora_targets(model) -> list[str]:
    # 针对 LLaMA/Transformer 结构（LFM2很可能基于此）常见的 LoRA 目标
    targets = set()
    for name, module in model.named_modules():
        # 针对 Attention 层的 q/k/v 投影
        if "q_proj" in name or "v_proj" in name:
            targets.add(name)
        # 如果 LFM2 有独立的视觉/多模态适配器，也应加入
        if "vision_adapter" in name or "mm_projector" in name:
            targets.add(name)

    # 移除通常不需要微调的层
    banned_suffixes = ("lm_head", "embed_tokens")
    targets = {t for t in targets if not any(t.endswith(x) for x in banned_suffixes)}
    return sorted(targets)


print(f"LoRA Target Modules: {find_lfm2_lora_targets(model)}")

lora_config = LoraConfig(
    r=LORA_R,
    lora_alpha=32,
    lora_dropout=0.1,
    target_modules=find_lfm2_lora_targets(model),
    init_lora_weights="gaussian",
)

model = prepare_model_for_kbit_training(model)
model = get_peft_model(model, lora_config)

# Create PyTorch dataset
train_dataset = LlavaDataset2(datapath, split="train", sort_json_key=False)
val_dataset = LlavaDataset2(datapath, split="validation", sort_json_key=False)


# ------------------ Collate Fns (适配 LFM2/LLaVA) -------------------

def train_collate_fn(examples):
    texts = []
    images = []

    for image, prompt_text, ground_truth, _ in examples:
        images.append(image)

        # ⚠️ 适配 LFM2/LLaVA 风格的 SFT 文本格式
        # 假设 LFM2 遵循 LLaVA 的 SFT 格式：USER: <image> prompt\nASSISTANT: ground_truth
        # 您需要根据 LFM2 模型的实际格式调整此处的模板
        # 这里使用一个简单的拼接作为示例
        full_text = f"USER: <image>\n{prompt_text}\nASSISTANT: {ground_truth}{processor.tokenizer.eos_token}"
        texts.append(full_text)

    # 1. Tokenize and Process Images
    # AutoProcessor 会处理图像和文本，并根据模型需求生成 pixel_values 和 pixel_attention_mask
    batch = processor(
        text=texts,
        images=images,
        padding="longest",  # 使用 longest 确保 mask 维度正确
        truncation=True,
        max_length=MAX_LENGTH,
        return_tensors="pt"
    )

    labels = batch["input_ids"].clone()
    # 仅计算 assistant (即答案) 部分的 loss。
    # 因为 LFM2/LLaVA 等模型的 labels mask 逻辑通常在 training_step 中处理，
    # 这里我们只将 padding ID 设为 -100。
    labels[labels == processor.tokenizer.pad_token_id] = -100
    batch["labels"] = labels
    spatial_shapes = batch.pop("spatial_shapes", None)
    pixel_attention_mask = batch.pop("pixel_attention_mask", None)

    # ⚠️ 修复：返回 pixel_attention_mask (而不是 image_grid_thw)
    return (
        batch["input_ids"],
        batch["attention_mask"],
        batch["pixel_values"],
        pixel_attention_mask,  # 关键：确保拿到这个 mask
        batch["labels"],spatial_shapes)


def eval_collate_fn(examples):
    texts = []
    images = []
    answers = []

    for image, prompt_text, ground_truth, _ in examples:
        images.append(image)
        answers.append(str(ground_truth))

        # ⚠️ 适配 LFM2/LLaVA 风格的推理文本格式
        # 只需要 prompt，模型会生成 assistant 部分
        prompt_text = f"USER: <image>\n{prompt_text}\nASSISTANT:"
        texts.append(prompt_text)

    # 1. Tokenize and Process Images
    batch = processor(
        text=texts,
        images=images,
        return_tensors="pt",
        padding="longest"
    )

    spatial_shapes = batch.pop("spatial_shapes", None)
    pixel_attention_mask = batch.pop("pixel_attention_mask", None)

    # ⚠️ 修复：返回 pixel_attention_mask (而不是 image_grid_thw)
    return (
        batch["input_ids"],
        batch["attention_mask"],
        batch["pixel_values"],
        pixel_attention_mask,  # 关键：确保拿到这个 mask
        answers,spatial_shapes)


# Define PyTorch LightningModule
class LFM2PLModule(L.LightningModule):
    def __init__(self, config, processor, model):
        super().__init__()
        self.config = config
        self.processor = processor
        self.model = model
        self.batch_size = config.get("batch_size")

    def training_step(self, batch, batch_idx):
        # ⚠️ 修复：将 image_grid_thw 替换为 pixel_attention_mask
        input_ids, attention_mask, pixel_values, pixel_attention_mask, labels,spatial_shapes = batch
        self.model.train()

        # ⚠️ 修复：传入 pixel_attention_mask
        outputs = self.model(input_ids=input_ids,
                             attention_mask=attention_mask,
                             pixel_values=pixel_values,
                             pixel_attention_mask=pixel_attention_mask,  # 传入
                             spatial_shapes=spatial_shapes,
                             labels=labels
                             )
        loss = outputs.loss
        batch_size = input_ids.size(0)

        self.log("train_loss", loss, batch_size=batch_size, sync_dist=True)

        # ⚠️ 提示：如果要实现只计算 LLaVA/LFM2 答案部分的 loss，
        # 逻辑应该在 dataset 或 training_step 中加入对 labels 的 mask，
        # 确保只有 assistant 的 token 对应的 label 不为 -100。

        return loss

    def validation_step(self, batch, batch_idx, dataset_idx=0):
        # ⚠️ 修复：将 image_grid_thw 替换为 pixel_attention_mask
        input_ids, attention_mask, pixel_values, pixel_attention_mask, answers,spatial_shapes = batch
        self.model.eval()

        # ⚠️ 修复：传入 pixel_attention_mask
        generated_ids = self.model.generate(input_ids=input_ids,
                                            attention_mask=attention_mask,
                                            pixel_values=pixel_values,
                                            pixel_attention_mask=pixel_attention_mask,  # 传入
                                            spatial_shapes=spatial_shapes,
                                            max_new_tokens=20
                                            )

        predictions_ids_only = generated_ids[:, input_ids.size(1):]
        predictions = self.processor.batch_decode(predictions_ids_only, skip_special_tokens=True)

        scores = []
        for pred, answer in zip(predictions, answers):
            # 清理预测结果
            pred = pred.strip()
            pred_norm = pred.lower()
            answer_norm = answer.lower()

            scores.append(edit_distance(pred_norm, answer_norm) / max(len(pred_norm), len(answer_norm), 1))

            if self.config.get("verbose", False) and len(scores) == 1 and self.global_rank == 0:
                print(f"Prediction: {pred} (Normalized: {pred_norm})")
                print(f"    Answer: {answer} (Normalized: {answer_norm})")
                print(f" Normed ED: {scores[0]}")

        self.log("val_edit_distance", np.mean(scores), sync_dist=True)

        return scores

    # 保持 configure_optimizers, train_dataloader, val_dataloader 不变

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.config.get("lr"))
        return optimizer

    def train_dataloader(self):
        return DataLoader(train_dataset, collate_fn=train_collate_fn, batch_size=self.batch_size, shuffle=True,
                          num_workers=4)

    def val_dataloader(self):
        return DataLoader(val_dataset, collate_fn=eval_collate_fn, batch_size=self.batch_size, shuffle=False,
                          num_workers=4)


config = {"max_epochs": EPOCH,
          "check_val_every_n_epoch": 1,
          "gradient_clip_val": 1.0,
          "accumulate_grad_batches": 4,
          "lr": 2e-5,
          "batch_size": 1,
          "num_nodes": 1,
          "warmup_steps": 50,
          "result_path": "./LFM2_VL",
          "verbose": True,
          }

model_module = LFM2PLModule(config, processor, model)

# Define callbacks (保持 SaveToDiskCallback, early_stop_callback, checkpoint_callback 不变)
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


early_stop_callback = EarlyStopping(monitor="val_edit_distance", patience=3, verbose=False, mode="min")

checkpoint_callback = ModelCheckpoint(
    dirpath='./share/LFM2-VL/',
    filename='lfm2-vl-450m-lora-yesno-best',
    save_top_k=1,
    verbose=True,
    monitor='val_edit_distance',
    mode='min'
)

# Train!
trainer = L.Trainer(
    accelerator="gpu",
    devices=1,
    strategy='deepspeed_stage_2',
    max_epochs=config.get("max_epochs"),
    accumulate_grad_batches=config.get("accumulate_grad_batches"),
    check_val_every_n_epoch=config.get("check_val_every_n_epoch"),
    gradient_clip_val=config.get("gradient_clip_val"),
    precision="bf16-mixed",
    log_every_n_steps=2,
    limit_val_batches=5,
    num_sanity_val_steps=0,
    callbacks=[SaveToDiskCallback(), early_stop_callback]
)

trainer.fit(model_module)