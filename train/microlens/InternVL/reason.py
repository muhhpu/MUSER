# -*- coding: utf-8 -*-
# InternVL 训练脚本：Lightning + DeepSpeed + LoRA + [ANS]/[REASON] 双loss
# 修复点：
# 1) 兼容 images/pixel_values/num_patches_list/image_flags
# 2) PEFT 不再吞掉视觉参数
# 3) 注册 <image> 为 special token，并写入 model.img_context_token_id
# 4) Trainer 关闭 sanity check（num_sanity_val_steps=0）

import os, re, sys, logging, inspect
import torch, numpy as np
import lightning as L
from torch.utils.data import DataLoader
from nltk.metrics.distance import edit_distance
from peft import LoraConfig, prepare_model_for_kbit_training, get_peft_model
from peft.peft_model import PeftModel, PeftModelForCausalLM
from transformers import AutoTokenizer, AutoModel
from lightning.pytorch.callbacks import Callback, ModelCheckpoint, EarlyStopping
from lightning.pytorch.strategies import DeepSpeedStrategy

# ====== 数据集 ======
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from load_llava_dataset import LlavaDataset2

# ====== 基本参数 ======
MODEL_ID = "/home/team//MLLM-MSR-main/MLLM-MSR/train/InternVL2-1B"
SAVE_DIR = "/home/team//MLLM-MSR-main/MLLM-MSR/save/InternVL2/internvl2-lora-fixed"
EPOCH = 4
LORA_R = 16
MAX_LENGTH = 3072
GEN_MAX_NEW_TOKENS = 256

logging.getLogger("transformers").setLevel(logging.ERROR)

# ====== Tokenizer & Model ======
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True, use_fast=False)
model = AutoModel.from_pretrained(MODEL_ID, trust_remote_code=True,
                                  torch_dtype=torch.bfloat16, low_cpu_mem_usage=True)

if hasattr(tokenizer, "padding_side"):
    tokenizer.padding_side = "right"

# --- 注册文本类特殊 token ---
SPECIAL_TOKENS = ["[ANS]", "[REASON]"]
try:
    added = tokenizer.add_tokens(SPECIAL_TOKENS)
    if added and hasattr(model, "resize_token_embeddings"):
        model.resize_token_embeddings(len(tokenizer))
except Exception as e:
    print("[Warn] add_tokens/resize_token_embeddings failed:", e)

# --- 关键：注册图像占位符 token，并绑定到 model.img_context_token_id ---
IMG_TOKEN = "<image>"   # 你的 prompt 里已经用的就是这个占位符（见 collate） ①
if tokenizer.convert_tokens_to_ids(IMG_TOKEN) == tokenizer.unk_token_id:
    tokenizer.add_tokens([IMG_TOKEN])
    if hasattr(model, "resize_token_embeddings"):
        model.resize_token_embeddings(len(tokenizer))
img_tid = tokenizer.convert_tokens_to_ids(IMG_TOKEN)
# 某些 InternVL 分支在 generate() 内部使用 self.img_context_token_id
# 没有就补上：
if not hasattr(model, "img_context_token_id") or model.img_context_token_id is None:
    try:
        model.img_context_token_id = img_tid
    except Exception:
        # 有些是包在 .model / .language_model 里
        if hasattr(model, "model"):
            setattr(model.model, "img_context_token_id", img_tid)
        else:
            # 最后兜底：挂到对象上
            setattr(model, "img_context_token_id", img_tid)

# ====== LoRA ======
def find_lora_targets(m):
    t = []
    for n, md in m.named_modules():
        if isinstance(md, torch.nn.Linear) and any(k in n for k in ["mlp", "projector", "mm_projector", "visual_projector"]):
            t.append(n)
    return sorted(set(t))

lora_targets = find_lora_targets(model)
lora_cfg = LoraConfig(
    r=LORA_R, lora_alpha=32, lora_dropout=0.1,
    bias="none", target_modules=lora_targets,
    init_lora_weights="gaussian", task_type="CAUSAL_LM",
)
model = prepare_model_for_kbit_training(model)
model = get_peft_model(model, lora_cfg)
import torch
import torch.nn as nn
from functools import wraps
import inspect
from peft.peft_model import PeftModel, PeftModelForCausalLM

# ====== 修补：PEFT forward (同时保留视觉参数 + 支持位置 pixel_values) ======
@wraps(PeftModel.forward)
def patched_peft_forward(self, *args, **kwargs):
    """
    修复：
    1. InternVLChatModel.forward 要求 pixel_values 是第二个位置参数
    2. 保留视觉参数，不删除 pixel_values/images 等
    3. 自动转 input_ids.long()
    """
    # ---- 先把 input_ids 强制转 long，防止 DeepSpeed bf16 导致 embedding 报错 ----
    if "input_ids" in kwargs and isinstance(kwargs["input_ids"], torch.Tensor):
        if kwargs["input_ids"].dtype != torch.long:
            kwargs["input_ids"] = kwargs["input_ids"].long()
    elif len(args) > 0 and isinstance(args[0], torch.Tensor):
        if args[0].dtype != torch.long:
            args = list(args)
            args[0] = args[0].long()
            args = tuple(args)

    sig = inspect.signature(self.base_model.forward)
    params = list(sig.parameters.keys())

    # --- 保留视觉参数 ---
    allowed = set(params)
    allowed.update(["pixel_values", "images", "image_flags",
                    "num_patches_list", "image_sizes", "image_grid_thw"])
    for k in list(kwargs.keys()):
        if k not in allowed:
            kwargs.pop(k, None)

    # --- 如果 forward 的第二个参数是 pixel_values，就位置传参 ---
    if len(params) > 1 and params[1] == "pixel_values" and "pixel_values" in kwargs:
        pv = kwargs.pop("pixel_values")
        if len(args) == 0:
            # 没有传 input_ids，用 kwargs 里的 input_ids
            input_ids = kwargs.pop("input_ids", None)
            return self.base_model.forward(input_ids, pv, **kwargs)
        else:
            return self.base_model.forward(args[0], pv, **kwargs)

    # --- 默认情况 ---
    return self.base_model.forward(*args, **kwargs)


PeftModel.forward = patched_peft_forward
if 'PeftModelForCausalLM' in globals():
    PeftModelForCausalLM.forward = patched_peft_forward

print("[Patch] Unified PEFT forward: keeps visual kwargs + positional pixel_values + input_ids.long()")

# ====== 强力补丁：防止 embedding 再次出错 ======
class _EmbeddingLongGuard(nn.Module):
    def __init__(self, inner: nn.Module):
        super().__init__()
        self.inner = inner
    def forward(self, indices: torch.Tensor):
        if indices.dtype not in (torch.int32, torch.int64):
            indices = indices.long()
        return self.inner(indices)

def _guard_language_model_embeddings(m):
    lm = getattr(m, "language_model", None)
    if lm is None:
        return
    emb = lm.get_input_embeddings()
    if not isinstance(emb, _EmbeddingLongGuard):
        lm.set_input_embeddings(_EmbeddingLongGuard(emb))
        print("[Patch] Embedding guard installed: input_ids -> long() before lookup")

_guard_language_model_embeddings(model)



print("[Patch] PEFT forward patched: visual kwargs preserved")

# ====== 数据集路径 ======
train_dataset = LlavaDataset2(
    "/home/team//MLLM-MSR-main/MLLM-MSR/MicroLens-50k-training-recurrent-longshortpreference-reason-new",
    split="train", sort_json_key=False
)
val_dataset = LlavaDataset2(
    "/home/team//MLLM-MSR-main/MLLM-MSR/MicroLens-50k-training-recurrent-longshortpreference-reason-new",
    split="validation", sort_json_key=False
)

# ====== 图像预处理（单图/单样本） ======
import torchvision.transforms as T
from torchvision.transforms.functional import InterpolationMode
IMAGENET_MEAN, IMAGENET_STD = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
def build_transform(size=448):
    return T.Compose([
        T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
        T.Resize((size, size), interpolation=InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
    ])
def images_to_pixel_values(pil_images, image_size=448):
    t = build_transform(image_size)
    tens = [t(img) for img in pil_images]
    return torch.stack(tens, dim=0), [1 for _ in pil_images]  # 每样本单张图，num_patches=1

# ====== collate ======
def normalize_ans(x: str) -> str:
    s = (x or "").strip().lower()
    if s in ["yes", "y", "yeah", "yep", "true", "1"]:
        return "yes"
    if s in ["no", "n", "nope", "false", "0"]:
        return "no"
    return "no"

def train_collate_fn(examples):
    pil_images, texts = [], []
    for (image, prompt_text, ground_truth, truth_reason) in examples:
        pil_images.append(image.convert("RGB"))
        ans = normalize_ans(ground_truth)
        # ① 这里 prompt 已经包含 <image>；我们上面把它注册成了特殊 token
        texts.append(f"[INST] {IMG_TOKEN}\n{prompt_text} [/INST] [ANS] {ans} [REASON] {truth_reason}")  # :contentReference[oaicite:1]{index=1}

    batch = tokenizer(texts, padding=True, truncation=True, max_length=MAX_LENGTH, return_tensors="pt")
    labels = batch["input_ids"].clone()
    labels[labels == tokenizer.pad_token_id] = -100
    pixel_values, num_patches_list = images_to_pixel_values(pil_images)
    return batch["input_ids"], batch.get("attention_mask"), pixel_values, num_patches_list, labels

def eval_collate_fn(examples):
    pil_images, texts, answers = [], [], []
    for ex in examples:
        if len(ex) == 4:
            image, prompt_text, gt, reason = ex
        else:
            image, prompt_text, gt = ex; reason = ""
        pil_images.append(image.convert("RGB"))
        texts.append(f"[INST] {IMG_TOKEN}\n{prompt_text} [/INST] [ANS]")  # 与训练一致，包含 <image> :contentReference[oaicite:2]{index=2}
        answers.append({"answer": gt, "reason": reason})
    batch = tokenizer(texts, return_tensors="pt", padding=True)
    pixel_values, num_patches_list = images_to_pixel_values(pil_images)
    return batch["input_ids"], batch.get("attention_mask"), pixel_values, num_patches_list, answers

# ====== Lightning 模块 ======
class InternVL2PLModule(L.LightningModule):
    def __init__(self, config, tokenizer, model):
        super().__init__()
        self.config, self.tokenizer, self.model = config, tokenizer, model
        self.alpha = torch.nn.Parameter(torch.tensor(1.0))
        self.beta  = torch.nn.Parameter(torch.tensor(0.5))
        print("[Model check] forward args:", self.model.forward.__code__.co_varnames)
        # 若模型仍没读到 img_context_token_id，则显式打印提示
        print("[Model check] img_context_token_id:", getattr(self.model, "img_context_token_id", None))

    def training_step(self, batch, batch_idx):
        input_ids, attention_mask, pixel_values, num_patches_list, labels = batch
        input_ids = input_ids.to(torch.long)
        labels = labels.to(torch.long)
        if attention_mask is not None:
            attention_mask = attention_mask.to(torch.long)
        forward_args = tuple(self.model.forward.__code__.co_varnames)

        # --- 构造其余 kwargs（不含 pixel_values） ---
        fw_kwargs = {"labels": None}
        if "attention_mask" in forward_args and attention_mask is not None:
            fw_kwargs["attention_mask"] = attention_mask
        if "num_patches_list" in forward_args:
            fw_kwargs["num_patches_list"] = num_patches_list
        if "image_flags" in forward_args:
            fw_kwargs["image_flags"] = torch.tensor(num_patches_list, device=pixel_values.device).unsqueeze(-1)
        if "images" in forward_args:
            # 某些分支把视觉张量命名为 images（而不是 pixel_values）
            # 这种分支通常仍然把视觉张量当作“第二个位置参数”
            # 所以这里不放到 kwargs，仍用位置参数喂
            pass

        # --- 关键：用“位置参数”喂给模型 ---
        # 绝大多数 InternVLChatModel.forward 是：forward(self, input_ids, pixel_values, ...)
        # 所以我们用：self.model(input_ids, pixel_values, **fw_kwargs)
        self.model.train()
        # --- 确保 image_flags 不是 None ---
        image_flags = torch.ones((pixel_values.size(0), 1), device=pixel_values.device, dtype=torch.int32)
        out = self.model(input_ids, pixel_values, image_flags=image_flags, **fw_kwargs)

        logits = out.logits
        # 下面维持原状（计算 loss）
        shift_logits, shift_labels = logits[..., :-1, :], labels[..., 1:]
        loss_fct = torch.nn.CrossEntropyLoss(reduction="none", ignore_index=-100)
        token_losses = loss_fct(shift_logits.reshape(-1, shift_logits.size(-1)),
                                shift_labels.reshape(-1)).reshape(shift_labels.size())

        ans_id = self.tokenizer.convert_tokens_to_ids("[ANS]")
        reason_id = self.tokenizer.convert_tokens_to_ids("[REASON]")
        batch_answer_mask = torch.zeros_like(shift_labels, dtype=torch.float32)
        batch_reason_mask = torch.zeros_like(shift_labels, dtype=torch.float32)
        pad_id = self.tokenizer.pad_token_id

        for b_idx, lab in enumerate(labels.cpu().tolist()):
            lab = [x if x != -100 else pad_id for x in lab]

            def find_pos(arr, tok):
                try:
                    return arr.index(tok)
                except ValueError:
                    return -1

            ans_pos = find_pos(lab, ans_id)
            reason_pos = find_pos(lab, reason_id)
            if ans_pos != -1:
                end = reason_pos if reason_pos > ans_pos else len(lab)
                batch_answer_mask[b_idx, ans_pos + 1:end] = 1.0
            if reason_pos != -1:
                batch_reason_mask[b_idx, reason_pos + 1:] = 1.0

        eps = 1e-8
        answer_loss = (token_losses * batch_answer_mask).sum() / (batch_answer_mask.sum() + eps)
        reason_loss = (token_losses * batch_reason_mask).sum() / (batch_reason_mask.sum() + eps)
        loss = self.alpha * answer_loss + self.beta * reason_loss

        self.log_dict({"train_answer_loss": answer_loss,
                       "train_reason_loss": reason_loss,
                       "train_loss": loss,
                       "alpha": self.alpha, "beta": self.beta}, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        # 解包 & 类型修正
        input_ids, attention_mask, pixel_values, num_patches_list, answers = batch
        input_ids = input_ids.to(torch.long)
        if attention_mask is not None:
            attention_mask = attention_mask.to(torch.long)

        # InternVL 需要 image_flags，不然 forward 会出错
        image_flags = torch.ones(
            (pixel_values.size(0), 1),
            device=pixel_values.device,
            dtype=torch.int32
        )

        # 读取模型 forward 的参数列表，确定该传什么键
        forward_args = set(self.model.forward.__code__.co_varnames)

        # ---- 构造 generate 调用参数 ----
        gen = {"input_ids": input_ids, "max_new_tokens": GEN_MAX_NEW_TOKENS}

        if "images" in forward_args:
            gen["images"] = pixel_values
        else:
            gen["pixel_values"] = pixel_values

        if "num_patches_list" in forward_args:
            gen["num_patches_list"] = num_patches_list
        if "image_flags" in forward_args:
            gen["image_flags"] = image_flags
        if attention_mask is not None:
            gen["attention_mask"] = attention_mask

        # ---- 生成输出 ----
        self.model.eval()
        with torch.no_grad():
            generated = self.model.generate(**gen)

        # ---- 解码与评估 ----
        preds = self.tokenizer.batch_decode(generated, skip_special_tokens=False)

        def norm_ans(a):
            a = (a or "").lower()
            if "yes" in a:
                return "yes"
            if "no" in a:
                return "no"
            return a

        acc = 0
        for pred, gold in zip(preds, answers):
            m = re.search(r"\[ANS\]\s*(.*?)\s*(\[REASON\].*)?", pred, re.S)
            p_ans = norm_ans(m.group(1).strip() if m else "")
            g_ans = norm_ans(str(gold.get("answer", "")))
            if p_ans and g_ans and (p_ans == g_ans):
                acc += 1

        self.log("val_acc", acc / max(1, len(preds)), prog_bar=True)

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.config.get("lr", 2e-5))

    def train_dataloader(self):
        return DataLoader(train_dataset, batch_size=1, shuffle=True, num_workers=4, collate_fn=train_collate_fn)

    def val_dataloader(self):
        return DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=4, collate_fn=eval_collate_fn)

# ====== Trainer ======
config = {"lr": 2e-5, "max_epochs": EPOCH}
module = InternVL2PLModule(config, tokenizer, model)
trainer = L.Trainer(
    accelerator="gpu",
    devices=1,
    precision="bf16-mixed",
    strategy=DeepSpeedStrategy(config={"zero_optimization": {"stage": 2}, "bf16": {"enabled": True}}),
    max_epochs=EPOCH,
    gradient_clip_val=1.0,
    log_every_n_steps=10,
    # 关闭 sanity check，避免还没 warmup 就先跑 generate():
    num_sanity_val_steps=0,
    callbacks=[EarlyStopping(monitor="val_acc", patience=3, mode="max")]
)

if __name__ == "__main__":
    trainer.fit(module)
