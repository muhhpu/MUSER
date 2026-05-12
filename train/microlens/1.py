# from load_llava_dataset import LlavaDataset, LlavaDataset2
#
# train_dataset = LlavaDataset2("/home/team//MLLM-MSR-main/MLLM-MSR/MicroLens-50k-training-recurrent-longshortpreference-reason",  split="train", sort_json_key=False)
# val_dataset = LlavaDataset2("/home/team//MLLM-MSR-main/MLLM-MSR/MicroLens-50k-training-recurrent-longshortpreference-reason", split="validation", sort_json_key=False)
#
# if len(train_dataset) > 0:
#     sample = train_dataset[0]
#     print("\n第一个训练样本的结构:")
#     print(f"类型: {type(sample)}")
#
#     if isinstance(sample, dict):
#         for key, value in sample.items():
#             print(f"{key}: {type(value)} - 示例: {str(value)[:100]}...")
#     else:
#         print(f"样本内容: {sample}")


# /home/team//MLLM-MSR-main/MLLM-MSR 快速检查:
# from load_llava_dataset import LlavaDataset2
# ds = LlavaDataset2("MicroLens-50k-training-recurrent-longshortpreference-reason-new-numeric",
#                    split="train", sort_json_key=False)
# sample = ds[0]
# img, prompt_text, gt = sample[0], sample[1], sample[2]
# print(type(img), getattr(img, "size", None), prompt_text[:80], gt)

from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
from load_llava_dataset import LlavaDataset2

MODEL_ID = "/home/team//MLLM-MSR-main/MLLM-MSR/train/Qwen2.5-VL-3B-Instruct"
processor = AutoProcessor.from_pretrained(MODEL_ID, trust_remote_code=True)
model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    MODEL_ID, torch_dtype="auto", device_map="auto", trust_remote_code=True
)

ds = LlavaDataset2("MicroLens-50k-training-recurrent-longshortpreference-reason-new-numeric",
                   split="validation", sort_json_key=False)
img, prompt_text, *_ = ds[0]

messages = [{"role":"user","content":[{"type":"image","image":img},{"type":"text","text":prompt_text}]}]
text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
inputs = processor(text=[text], images=[[img]], padding=True, return_tensors="pt").to(model.device)

out_ids = model.generate(**inputs, max_new_tokens=64)
print(processor.batch_decode(out_ids, skip_special_tokens=True)[0])

