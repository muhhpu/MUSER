import torch
from multiprocess import set_start_method
from transformers import AutoProcessor, LlavaNextForConditionalGeneration
from datasets import load_dataset
from PIL import ImageOps
from torch.cuda.amp import autocast
import os
import pandas as pd

# 环境设置
os.environ['CURL_CA_BUNDLE'] = ''
os.environ["CUDA_VISIBLE_DEVICES"] = "2,3,4,5,6,7"

# ==== 模型加载 ====
model_id = "/home/team//llava-v1.6-mistral-7b-hf"

# 判断是否可用 flash_attention_2
attn_mode = "sdpa"
try:
    import flash_attn
    attn_mode = "flash_attention_2"
    print("[INFO] 使用 FlashAttention 2")
except ImportError:
    print("[WARNING] 未检测到 FlashAttention 2，将使用 sdpa 注意力实现")

model = LlavaNextForConditionalGeneration.from_pretrained(
    model_id,
    cache_dir='/data1/share/.HF_cache/',
    attn_implementation=attn_mode,
    torch_dtype=torch.float16,
    device_map="auto"
).eval()

processor = AutoProcessor.from_pretrained(model_id)

prompt = (
    "[INST] <image>\nPlease describe this image, which is a cover of a video."
    " Provide a detailed description in one continuous paragraph, including content information and visual features such as colors, objects, text,"
    " and any notable elements present in the image.[/INST]"
)

# ==== 数据集处理 ====
def add_image_file_path(example):
    file_path = example['image'].filename
    filename = os.path.splitext(os.path.basename(file_path))[0]
    example['item_id'] = filename
    return example

img_dir = "/home/team//MicroLens-50k-Dataset/MicroLens-50k_covers"
dataset = load_dataset("imagefolder", data_dir=img_dir)
dataset = dataset.map(add_image_file_path)
print(dataset)

# ==== GPU 批处理推理 ====
def gpu_computation(batch, rank):
    device = f"cuda:{(rank or 0) % torch.cuda.device_count()}"
    model.to(device)

    batch_images = [img for img in batch['image'] if img is not None]  # 去除 None

    # 对齐尺寸
    max_width = max(img.width for img in batch_images)
    max_height = max(img.height for img in batch_images)
    padded_images = []
    for img in batch_images:
        if img.width != max_width or img.height != max_height:
            delta_width = max_width - img.width
            delta_height = max_height - img.height
            padding = (
                delta_width // 2,
                delta_height // 2,
                delta_width - (delta_width // 2),
                delta_height - (delta_height // 2)
            )
            img = ImageOps.expand(img, border=padding, fill='black')
        padded_images.append(img)

    # 模型输入
    model_inputs = processor(
        [prompt] * len(padded_images),
        padded_images,
        return_tensors="pt",
        padding=True
    ).to(device, torch.float16)

    # 推理
    with torch.no_grad():
        with autocast(device_type="cuda", dtype=torch.float16):
            outputs = model.generate(**model_inputs, max_new_tokens=200)

    ans = processor.batch_decode(outputs, skip_special_tokens=True)
    ans = [a.split("[/INST]")[-1] for a in ans]
    return {"summary": ans}

# ==== 主程序 ====
if __name__ == "__main__":
    set_start_method("spawn")
    updated_dataset = dataset.map(
        gpu_computation,
        batched=True,
        batch_size=8,
        with_rank=True,
        num_proc=4
    )

    train_dataset = updated_dataset['train']
    df = pd.DataFrame({
        'item_id': train_dataset['item_id'],
        'summary': train_dataset['summary']
    })
    df.to_csv('image_summary.csv', index=False)
