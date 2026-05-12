# -*- coding: utf-8 -*-
"""
Qwen2.5-VL SFT (messages + image) minimal training script with LoRA and Lightning.
- Keeps your dataset reading style (LlavaDataset3) and example tuple format.
- Builds inputs via messages -> apply_chat_template -> processor(..., images=[[PIL.Image]]).
- Masks labels so only tokens from [ANS] and [REASON] onward are trained (as in your previous logic).
- Uses Flash-Attn2 if available; set --no_fa2 to disable.
- Supports 4bit QLoRA via --qlora.
"""
import multiprocessing
multiprocessing.set_start_method("spawn", force=True)
import os
import re
import math
import argparse
import sys
from typing import List, Tuple, Any

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

import pytorch_lightning as pl
from pytorch_lightning.utilities.rank_zero import rank_zero_only
from pytorch_lightning.strategies import DDPStrategy

from transformers import (
    AutoProcessor,
    Qwen2_5_VLForConditionalGeneration,
    BitsAndBytesConfig,
    get_cosine_schedule_with_warmup,
)

# 你仓库里的数据集类（保持不动）
# 期望 __getitem__ 返回: (PIL.Image, prompt_text, ground_truth[, truth_reason])


from load_llava_dataset import LlavaDataset3

# qwen 辅助视觉解析（可选）
try:
    from qwen_vl_utils import process_vision_info
    _HAS_QWEN = True
except Exception:
    _HAS_QWEN = False


SPECIAL_TOKENS = ["[ANS]", "[REASON]"]

# ---- 新增：可 pickling 的 collator ----
class Collator:
    def __init__(self, model_path: str, max_length: int, is_train: bool):
        self.model_path = model_path
        self.max_length = max_length
        self.is_train = is_train
        self._processor = None  # 在 worker 内懒加载

    def _ensure_processor(self):
        if self._processor is None:
            from transformers import AutoProcessor
            self._processor = AutoProcessor.from_pretrained(
                self.model_path, trust_remote_code=True
            )
            self._processor.tokenizer.padding_side = "right"
            if (self._processor.tokenizer.pad_token_id is None and
                self._processor.tokenizer.eos_token_id is not None):
                self._processor.tokenizer.pad_token = self._processor.tokenizer.eos_token

    def __call__(self, examples):
        self._ensure_processor()
        if self.is_train:
            return train_collate_fn(examples, self._processor, self.max_length)
        else:
            return eval_collate_fn(examples, self._processor)

def normalize_ans(x: str) -> str:
    x = str(x).strip().lower()
    if x in ["yes", "y", "1", "true"]:
        return "yes"
    if x in ["no", "n", "0", "false"]:
        return "no"
    # 兜底：若标签不是 yes/no，就原样返回
    return x


class QwenVLDataModule(pl.LightningDataModule):
    def __init__(self, data_root: str, processor, batch_size: int = 2, num_workers: int = 4,
                 max_length: int = 2048,model_path: str = None):
        super().__init__()
        self.data_root = data_root
        self.processor = processor
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.max_length = max_length
        self.model_path = model_path  # ⭐ 记得存一下

    def setup(self, stage=None):
        # 你之前的命名：train / validation
        self.train_set = LlavaDataset3(self.data_root, split="train", sort_json_key=False)
        try:
            self.val_set = LlavaDataset3(self.data_root, split="validation", sort_json_key=False)
        except Exception:
            self.val_set = None

    def train_dataloader(self):
        return DataLoader(
            self.train_set,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            collate_fn=Collator(self.model_path, self.max_length, is_train=True),
            pin_memory=True,
            drop_last=True,
        )

    def val_dataloader(self):
        if self.val_set is None:
            return None
        return DataLoader(
            self.val_set,
            batch_size=1,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=Collator(self.model_path, self.max_length, is_train=False),
            pin_memory=True,
            drop_last=False,
        )


def build_messages_for_train(image, prompt_text: str, ans: str, reason: str):
    """user: image + text; assistant: labeled answer+reason"""
    return [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text",  "text": f"{prompt_text} \n [ANS] {ans} \n [REASON] {reason}"},
            ],
        },
    ]


def build_messages_for_infer(image, prompt_text: str):
    return [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text",  "text": prompt_text},
            ],
        }
    ]


def mask_labels_keep_from_ans(input_ids: torch.Tensor,
                              tokenizer,
                              pad_token_id: int) -> torch.Tensor:
    """
    将 label 置为 input_ids 并在 [ANS] 第一次出现之前全置为 -100。
    这样 loss 仅覆盖 assistant 的标注段（含 [ANS]/[REASON]）。
    """
    labels = input_ids.clone()
    labels[labels == pad_token_id] = -100

    # 定位 [ANS] 的起始位置
    ans_tok = tokenizer.convert_tokens_to_ids("[ANS]")
    # 小心：若 tokenizer 不把 "[ANS]" 当作单个 token，可退化到基于字符串再 tokenize 的方案。
    # 这里先做一次快速路径，如果没命中再做字符串级别查找。
    for i in range(labels.size(0)):
        row = labels[i]
        idx = (row == ans_tok).nonzero(as_tuple=True)[0] if ans_tok is not None else torch.tensor([])
        if idx.numel() > 0:
            start = idx[0].item()
            # [ANS] 之前置 -100
            if start > 0:
                row[:start] = -100
        else:
            # 回退方案：用解码粗略查找位置（性能一般，但小 batch 可接受）
            text = tokenizer.decode(input_ids[i].tolist(), skip_special_tokens=False)
            pos = text.find("[ANS]")
            if pos > 0:
                # 重新 tokenize 到 pos 的字符串长度，得出 token 边界大致位置
                prefix = text[:pos]
                approx = tokenizer(prefix, add_special_tokens=False, return_tensors="pt").input_ids[0]
                start = min(len(approx), input_ids.size(1)-1)
                row[:start] = -100
        labels[i] = row
    return labels




def train_collate_fn(batch,processor):
    messages = []
    images = []
    videos = []

    for sample in batch:
        image = sample["image"]
        prompt_text = sample["prompt"]
        ans = sample["ground_truth"]
        truth_reason = sample["truth_reason"]

        message = build_messages_for_train(image, prompt_text, ans, truth_reason)

        messages.append(message)
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    # ✅ 关键：用官方 process_vision_info 拿图像与 grid
    images, videos = process_vision_info(messages)
    inputs = processor(
        text=[text],
        images=images,
        videos=[],
        padding=True,
        truncation=True,
        return_tensors="pt",
    )

    image_grid_thw = getattr(inputs, "image_grid_thw", None)
    if image_grid_thw is None:
        image_grid_thw = inputs.get("image_grid_thw", None)

    labels = mask_labels_keep_from_ans(
        inputs["input_ids"], processor.tokenizer, processor.tokenizer.pad_token_id
    )

    return inputs["input_ids"], inputs["attention_mask"], inputs.get("pixel_values", None), image_grid_thw, labels




def eval_collate_fn(batch, processor):
    messages = []
    images = []
    videos = []

    for sample in batch:
        image = sample["image"]
        prompt_text = sample["prompt"]
        ans = sample["ground_truth"]
        truth_reason = sample["truth_reason"]

        message = build_messages_for_train(image, prompt_text, ans, truth_reason)
        messages.append(message)

    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    # ✅ 关键：用官方 process_vision_info 拿图像与 grid
    images, videos = process_vision_info(messages)
    inputs = processor(
        text=[text],
        images=images,
        videos=[],
        padding=True,
        truncation=True,
        return_tensors="pt",
    )

    image_grid_thw = getattr(inputs, "image_grid_thw", None)
    if image_grid_thw is None:
        image_grid_thw = inputs.get("image_grid_thw", None)

    labels = mask_labels_keep_from_ans(
        inputs["input_ids"], processor.tokenizer, processor.tokenizer.pad_token_id
    )

    return inputs["input_ids"], inputs["attention_mask"], inputs.get("pixel_values", None), image_grid_thw, labels



class LitQwenSFT(pl.LightningModule):
    def __init__(self, model, processor, lr=2e-5, wd=0.0, warmup_ratio=0.03, max_steps=None):
        super().__init__()
        self.model = model
        self.processor = processor
        self.save_hyperparameters(ignore=["model", "processor"])

    def training_step(self, batch, batch_idx):
        input_ids, attention_mask, pixel_values, image_grid_thw, labels = batch
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            pixel_values=pixel_values,
            image_grid_thw=image_grid_thw,  # ⭐ 关键
            labels=labels,
        )
        loss = outputs.loss
        self.log("train/loss", loss, prog_bar=True, on_step=True, on_epoch=True)
        return loss

    @torch.no_grad()
    def validation_step(self, batch, batch_idx):
        input_ids, attention_mask, pixel_values, image_grid_thw, golds = batch
        print("🔍 image_grid_thw:", image_grid_thw)
        print("pixel_values:", None if pixel_values is None else pixel_values.shape)
        gen_ids = self.model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            pixel_values=pixel_values,
            image_grid_thw=image_grid_thw,  # ⭐ 关键
            max_new_tokens=128,
        )

        # 直接解码整句（包含用户模板）；更严格的话可以裁去输入长度
        out = self.processor.batch_decode(gen_ids, skip_special_tokens=True)
        # 简单地把首行打印出来做 sanity check
        self.log("val/samples", 0.0, prog_bar=False)  # 占位，防报错
        if batch_idx == 0 and self.global_rank == 0:
            print("\n[VAL SAMPLE OUTPUT]\n", out[0][:400])
        # 这里你可以把 out 与 golds["answer"] 做匹配计算 AUC/Recall@K 等，你之前有评估脚本的话沿用即可。

    def configure_optimizers(self):
        no_decay = ["bias", "LayerNorm.weight", "layer_norm.weight"]
        decay, nodecay = [], []
        for n, p in self.named_parameters():
            if any(nd in n for nd in no_decay):
                nodecay.append(p)
            else:
                decay.append(p)
        optim = torch.optim.AdamW(
            [{"params": decay, "weight_decay": self.hparams.wd},
             {"params": nodecay, "weight_decay": 0.0}],
            lr=self.hparams.lr,
            betas=(0.9, 0.999),
            eps=1e-8
        )

        # 若未设置 max_steps，就用 epoch*steps 的方式
        if self.trainer.max_steps and self.trainer.max_steps > 0:
            t_total = self.trainer.max_steps
        else:
            # 估算一个 steps（不精确但够用）
            train_loader = self.trainer.datamodule.train_dataloader()
            steps_per_epoch = math.ceil(len(train_loader.dataset) / (train_loader.batch_size))
            t_total = steps_per_epoch * max(1, self.trainer.max_epochs)

        warmup_steps = int(self.hparams.warmup_ratio * t_total)
        sched = get_cosine_schedule_with_warmup(optim, warmup_steps, t_total)
        return {
            "optimizer": optim,
            "lr_scheduler": {"scheduler": sched, "interval": "step", "frequency": 1}
        }


def find_lora_targets_qwen(model, start_layer=0, end_layer=None, include_projector=True) -> List[str]:
    """鲁棒选择 Qwen 的 MLP 线性层 + 可选多模态投影层"""
    linear = nn.Linear
    # 推断层数与前缀
    layer_ids = set()
    prefixes = set()
    for name, _ in model.named_modules():
        m = re.search(r"\.layers\.(\d+)\.", name)
        if m:
            layer_ids.add(int(m.group(1)))
            prefixes.add(name[:m.start()])
    if not layer_ids:
        num_layers = 32
        prefixes = {"model"}
    else:
        num_layers = max(layer_ids) + 1
    if end_layer is None:
        end_layer = num_layers - 1

    wanted = ("mlp.gate_proj", "mlp.up_proj", "mlp.down_proj", "mlp.w1", "mlp.w2", "mlp.w3")
    banned = ("attn.", "self_attn.", "embed_tokens", "lm_head")

    targets = set()
    if include_projector:
        for n, m in model.named_modules():
            if "multi_modal_projector" in n and isinstance(m, linear):
                targets.add(n)

    for lid in range(start_layer, end_layer + 1):
        for pref in prefixes:
            root = f"{pref}.layers.{lid}."
            for n, m in model.named_modules():
                if n.startswith(root) and isinstance(m, linear):
                    if any(n.endswith(w) for w in wanted) and not any(b in n for b in banned):
                        targets.add(n)
    return sorted(list(targets))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True,
                        help="Local path to Qwen2.5-VL-3B-Instruct")
    parser.add_argument("--data_root", type=str, required=True)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--max_length", type=int, default=2048)
    parser.add_argument("--max_epochs", type=int, default=1)
    parser.add_argument("--precision", type=str, choices=["bf16", "fp16"], default="bf16")
    parser.add_argument("--accumulate", type=int, default=8)
    parser.add_argument("--start_lora_layer", type=int, default=16)
    parser.add_argument("--end_lora_layer", type=int, default=-1)
    parser.add_argument("--qlora", action="store_true")
    parser.add_argument("--no_fa2", action="store_true")
    args = parser.parse_args()

    dtype = torch.bfloat16 if args.precision == "bf16" else torch.float16

    # Processor
    processor = AutoProcessor.from_pretrained(args.model_path, trust_remote_code=True)
    # 强制单图 pad 模式，避免拆成多块导致 grid 不匹配
    try:
        ip = processor.image_processor
        if hasattr(ip, "image_aspect_ratio"):
            ip.image_aspect_ratio = "pad"
        if hasattr(ip, "do_dynamic_resize"):
            ip.do_dynamic_resize = False
        if hasattr(ip, "size") and isinstance(ip.size, dict):
            ip.size["shortest_edge"] = 448
    except Exception:
        pass

    # tokenizer padding
    processor.tokenizer.padding_side = "right"
    if processor.tokenizer.pad_token_id is None and processor.tokenizer.eos_token_id is not None:
        processor.tokenizer.pad_token = processor.tokenizer.eos_token

    # add special tokens
    added = processor.tokenizer.add_tokens(SPECIAL_TOKENS)

    # Model
    if args.qlora:
        bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=dtype)
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            args.model_path, torch_dtype=dtype,
            quantization_config=bnb, trust_remote_code=True,
        )
    else:
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            args.model_path, torch_dtype=dtype,
            attn_implementation=None if args.no_fa2 else "flash_attention_2",
            low_cpu_mem_usage=True, trust_remote_code=True,
        )

    if added and added > 0:
        model.resize_token_embeddings(len(processor.tokenizer))

    # --- LoRA ---
    from peft import LoraConfig, prepare_model_for_kbit_training, get_peft_model
    if args.end_lora_layer < 0:
        args.end_lora_layer = None
    target_modules = find_lora_targets_qwen(
        model, start_layer=args.start_lora_layer, end_layer=args.end_lora_layer, include_projector=True
    )
    lcfg = LoraConfig(
        r=8, lora_alpha=32, lora_dropout=0.1, bias="none", target_modules=target_modules, task_type="CAUSAL_LM"
    )
    if args.qlora:
        model = prepare_model_for_kbit_training(model)
    model = get_peft_model(model, lcfg)

    # Data
    dm = QwenVLDataModule(
        data_root=args.data_root, processor=processor,
        batch_size=args.batch_size, num_workers=args.num_workers, max_length=args.max_length,model_path=args.model_path
    )
    dm.setup()

    # Lightning
    precision = "bf16-mixed" if args.precision == "bf16" else "16-mixed"
    strategy = DDPStrategy(find_unused_parameters=False) if torch.cuda.device_count() > 1 else "auto"

    lit = LitQwenSFT(
        model=model, processor=processor, lr=args.lr, wd=args.weight_decay, warmup_ratio=0.03
    )

    trainer = pl.Trainer(
        accelerator="gpu",
        devices=torch.cuda.device_count(),
        strategy=strategy,
        precision=precision,
        max_epochs=args.max_epochs,
        accumulate_grad_batches=args.accumulate,
        gradient_clip_val=1.0,
        log_every_n_steps=10,
        enable_checkpointing=False,
    )

    trainer.fit(lit, datamodule=dm)


if __name__ == "__main__":
    main()
