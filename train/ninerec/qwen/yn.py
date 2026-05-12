# ------------------ Qwen 特有导入 -------------------
from transformers import AutoProcessor, AutoTokenizer, Qwen2_5_VLForConditionalGeneration
from qwen_vl_utils import process_vision_info
# --------------------------------------------------
from transformers import BitsAndBytesConfig
import torch
from torch.utils.data import Dataset
from typing import Any, Dict
import random
from peft import LoraConfig, prepare_model_for_kbit_training, get_peft_model
from load_llava_dataset import LlavaDataset, LlavaDataset2  # 确保这个文件在
from PIL import ImageOps
from lightning.pytorch.loggers import WandbLogger
import lightning as L
from torch.utils.data import DataLoader
import re
import os
from nltk import edit_distance
import numpy as np
from lightning.pytorch.callbacks import Callback, ModelCheckpoint
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
import logging
from huggingface_hub import HfApi

# ------------------ 常量配置 (已更新) -------------------
MAX_LENGTH = 4096
EPOCH = 2
LORA_R = 8

# ⚠️ 更新为你的 Qwen2.5-VL 模型路径
MODEL_ID = "/home/team//MLLM-MSR-main/MLLM-MSR/train/Qwen2.5-VL-3B-Instruct/"

prompt_template = "yesno"

REPO_ID = "yeyuyang95/qwen2.5-vl-3b-lora"  # 建议修改
WANDB_PROJECT = "Qwen2.5-VL"  # 建议修改
WANDB_NAME = f"qwen2.5-vl-3b-lora-{prompt_template}"  # 建议修改

# ⚠️ 更新保存路径以反映新模型
SAVE_DIR = f"/home/team//MLLM-MSR-main/MLLM-MSR/save/Qwen2.5-VL/ninerec-qwen2.5-vl-3b-lora-recurrent-user-noLS-{prompt_template}-e{EPOCH}-r{LORA_R}"

# ⚠️ 确保这个数据集里 'ground_truth' 是 'yes'/'no'
datapath = f"ninerec-training-recurrent-noLS-yesno-imagepath"

logging.getLogger("transformers").setLevel(logging.ERROR)

processor = AutoProcessor.from_pretrained(MODEL_ID)
processor.tokenizer.padding_side = "right"

# ⚠️ 修复：Qwen 也需要添加特殊 token
SPECIAL_TOKENS = ["[ANS]", "[REASON]"]  # 即使只用 "yes/no", 如果 prompt 里有也需要
added = processor.tokenizer.add_tokens(SPECIAL_TOKENS)

USE_LORA = True
USE_QLORA = False

## Load model
if USE_QLORA or USE_LORA:
    if USE_QLORA:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.float16
        )
        # ⚠️ 已修改为 Qwen
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            MODEL_ID,
            torch_dtype=torch.float16,
            quantization_config=bnb_config,

        )
    else:
        # ⚠️ 已修改为 Qwen
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            MODEL_ID,
            torch_dtype=torch.float16,
            _attn_implementation="flash_attention_2",

        )
else:
    # ⚠️ 已修改为 Qwen
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.float16,
        _attn_implementation="flash_attention_2",

    )

# ⚠️ 修复：如果添加了 token，需要 resize
if added and added > 0:
    model.resize_token_embeddings(len(processor.tokenizer))


# Apply PEFT
def find_reasoning_lora_targets(
        model,
        start_layer: int = 0,
        end_layer: int | None = None,
        include_projector: bool = True,
) -> list[str]:
    linear_cls = torch.nn.Linear
    targets = set()
    try:
        num_layers = len(model.language_model.layers)
    except Exception:
        layer_ids = []
        for name, _ in model.named_modules():
            m = re.search(r"language_model\.layers\.(\d+)\.", name)
            if m:
                layer_ids.append(int(m.group(1)))
        num_layers = (max(layer_ids) + 1) if layer_ids else 32
    if end_layer is None:
        end_layer = num_layers - 1

    if include_projector:
        for name, module in model.named_modules():
            if "vision_adapter" in name and isinstance(module, linear_cls):
                targets.add(name)

    wanted_ffn_suffixes = ("mlp.gate_proj", "mlp.up_proj", "mlp.down_proj")
    for lid in range(start_layer, end_layer + 1):
        prefix = f"model.language_model.layers.{lid}."
        for suffix in wanted_ffn_suffixes:
            full = prefix + suffix
            try:
                mod = dict(model.named_modules())[full]
                if isinstance(mod, linear_cls):
                    targets.add(full)
            except KeyError:
                for name, module in model.named_modules():
                    if name.startswith(prefix) and name.endswith(suffix) and isinstance(module, linear_cls):
                        targets.add(name)

    banned_suffixes = (
        "self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.o_proj",
        "embed_tokens", "rotary_emb", "lm_head",
        "input_layernorm", "post_attention_layernorm", "layer_norm", "layernorm",
    )
    targets = {t for t in targets if not any(t.endswith(x) for x in banned_suffixes)}
    return sorted(targets)


print(f"LoRA Target Modules: {find_reasoning_lora_targets(model)}")

lora_config = LoraConfig(
    r=LORA_R,
    lora_alpha=32,
    lora_dropout=0.1,
    target_modules=find_reasoning_lora_targets(model),
    init_lora_weights="gaussian",
)

model = prepare_model_for_kbit_training(model)
model = get_peft_model(model, lora_config)

# Create PyTorch dataset
train_dataset = LlavaDataset2(datapath, split="train", sort_json_key=False)
val_dataset = LlavaDataset2(datapath, split="validation", sort_json_key=False)


# ⚠️ 已移除: resize_image 函数

# ------------------ Collate Fns (已适配 Qwen) -------------------

def train_collate_fn(examples):
    all_messages = []
    for example in examples:
        # LlavaDataset2 返回: image, prompt_text, ground_truth, _ (truth_reason)
        image, prompt_text, ground_truth,_ = example

        # 构建 Qwen message 格式
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image,"max_pixels": 1280 * 1280},
                    {"type": "text", "text": prompt_text}
                ]
            },
            {
                "role": "assistant",
                "content": ground_truth  # 直接是 "yes" 或 "no"
            }
        ]
        all_messages.append(messages)

    # 1. 使用 chat_template (SFT 时 add_generation_prompt=False)
    texts = [
        processor.apply_chat_template(msg, tokenize=False, add_generation_prompt=False)
        for msg in all_messages
    ]

    # 2. 使用 qwen_vl_utils 提取图像
    image_inputs, video_inputs = process_vision_info(all_messages)

    # 3. Tokenize
    batch = processor(
        text=texts,
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        truncation=True,
        max_length=MAX_LENGTH,
        return_tensors="pt"
    )

    labels = batch["input_ids"].clone()
    labels[labels == processor.tokenizer.pad_token_id] = -100
    batch["labels"] = labels

    # ⚠️ 修复: 增加 image_grid_thw
    return (
    batch["input_ids"], batch["attention_mask"], batch["pixel_values"], batch.get("image_grid_thw"), batch["labels"])


def eval_collate_fn(examples):
    all_messages = []
    answers = []

    for example in examples:
        image, prompt_text, ground_truth,_ = example

        # 构建 Qwen message 格式 (只有 user)
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
        answers.append(ground_truth)

    # 1. 使用 chat_template (推理时 add_generation_prompt=True)
    texts = [
        processor.apply_chat_template(msg, tokenize=False, add_generation_prompt=True)
        for msg in all_messages
    ]

    # 2. 使用 qwen_vl_utils 提取图像
    image_inputs, video_inputs = process_vision_info(all_messages)

    # 3. Tokenize
    batch = processor(
        text=texts,
        images=image_inputs,
        videos=video_inputs,
        return_tensors="pt",
        padding=True
    )

    # ⚠️ 修复: 增加 image_grid_thw
    return (batch["input_ids"], batch["attention_mask"], batch["pixel_values"], batch.get("image_grid_thw"), answers)


# Define PyTorch LightningModule
class QwenModelPLModule(L.LightningModule):  # 名字改一下
    def __init__(self, config, processor, model):
        super().__init__()
        self.config = config
        self.processor = processor
        self.model = model
        self.batch_size = config.get("batch_size")

    def training_step(self, batch, batch_idx):
        # ⚠️ 修复: 增加 image_grid_thw
        input_ids, attention_mask, pixel_values, image_grid_thw, labels = batch
        self.model.train()

        # ⚠️ 修复: 传入 image_grid_thw
        outputs = self.model(input_ids=input_ids,
                             attention_mask=attention_mask,
                             pixel_values=pixel_values,
                             image_grid_thw=image_grid_thw,  # 传入
                             labels=labels
                             )
        loss = outputs.loss
        batch_size = input_ids.size(0)

        self.log("train_loss", loss, batch_size=batch_size, sync_dist=True)

        return loss

    def validation_step(self, batch, batch_idx, dataset_idx=0):
        # ⚠️ 修复: 增加 image_grid_thw
        input_ids, attention_mask, pixel_values, image_grid_thw, answers = batch
        self.model.eval()

        # ⚠️ 修复: 传入 image_grid_thw
        generated_ids = self.model.generate(input_ids=input_ids,
                                            attention_mask=attention_mask,
                                            pixel_values=pixel_values,
                                            image_grid_thw=image_grid_thw,  # 传入
                                            max_new_tokens=20  # yes/no 预测不需要 MAX_LENGTH
                                            )

        # 截断 prompt 部分
        predictions_ids_only = generated_ids[:, input_ids.size(1):]
        # 解码
        predictions = self.processor.batch_decode(predictions_ids_only, skip_special_tokens=True)

        scores = []
        for pred, answer in zip(predictions, answers):
            # ⚠️ 修复: 使用 Qwen 的清理逻辑
            pred = pred.replace("<|im_end|>", "").strip()
            # 你的 ground_truth 可能是 'Yes' 或 'No'，也可能是 'yes'/'no'
            # 统一转小写比较
            pred_norm = pred.lower()
            answer_norm = answer.lower()

            scores.append(edit_distance(pred_norm, answer_norm) / max(len(pred_norm), len(answer_norm), 1))

            if self.config.get("verbose", False) and len(scores) == 1 and self.global_rank == 0:
                print(f"Prediction: {pred} (Normalized: {pred_norm})")
                print(f"    Answer: {answer} (Normalized: {answer_norm})")
                print(f" Normed ED: {scores[0]}")

        self.log("val_edit_distance", np.mean(scores), sync_dist=True)

        return scores

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
          "batch_size": 1,  # micro_batch_size
          "num_nodes": 1,
          "warmup_steps": 50,
          "result_path": "./Qwen_VL",  # 路径更新
          "verbose": True,
          }

model_module = QwenModelPLModule(config, processor, model)

# Define callbacks
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
    dirpath='./share/Qwen2.5-VL/',  # 路径更新
    filename='qwen2.5-vl-3b-lora-yesno-best',  # 名字更新
    save_top_k=1,
    verbose=True,
    monitor='val_edit_distance',  # 监控 val_edit_distance
    mode='min'
)

# Train!
trainer = L.Trainer(
    accelerator="gpu",
    devices=3,
    strategy='deepspeed_stage_2',
    max_epochs=config.get("max_epochs"),
    accumulate_grad_batches=config.get("accumulate_grad_batches"),
    check_val_every_n_epoch=config.get("check_val_every_n_epoch"),
    gradient_clip_val=config.get("gradient_clip_val"),
    precision="bf16-mixed",
    log_every_n_steps=2,
    limit_val_batches=5,
    num_sanity_val_steps=0,
    callbacks=[SaveToDiskCallback(), early_stop_callback]  # 添加了 early_stop 和 checkpoint
)

trainer.fit(model_module)