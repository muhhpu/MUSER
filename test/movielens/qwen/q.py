import torch
import os
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration, BitsAndBytesConfig
from peft import PeftModel
# 假设 qwen_vl_utils.py 包含 process_vision_info 函数，这里我们只需要导入它
# 如果你没有这个文件，可能需要从 Qwen-VL 的官方仓库中找到或创建一个空的占位函数
try:
    from qwen_vl_utils import process_vision_info
except ImportError:
    print("Warning: qwen_vl_utils not found. Using a placeholder for process_vision_info.")
    def process_vision_info(messages):
        # 对于纯文本问题，不需要处理视觉信息，返回 None
        return None, None

# --- 配置 ---
# 设置环境变量，固定使用 CUDA 设备 (假设你希望使用 CUDA)
os.environ['CURL_CA_BUNDLE'] = ''
os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # 假设使用 CUDA 0
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# 模型和路径配置 (与你的 1.py 文件一致)
BASE_MODEL_ID = "/home/team//MLLM-MSR-main/MLLM-MSR/train/Qwen2.5-VL-3B-Instruct/"
PEFT_MODEL_ID = "/home/team//MLLM-MSR-main/MLLM-MSR/save/Qwen2.5-VL/qwen2.5-vl-3b-lora-recurrent-user-longshort-finetunereason-kua-linear1-e4-r8-new-llm全layer-numeric"
QUESTION = "2025是平年还是闰年？"

# --- 1. 加载基座模型和处理器 ---
print(f"Loading base model from: {BASE_MODEL_ID}")
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    trust_remote_code=True
)

base_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    BASE_MODEL_ID,
    quantization_config=bnb_config,
    device_map="auto",
    trust_remote_code=True,
)

processor = AutoProcessor.from_pretrained(BASE_MODEL_ID, trust_remote_code=True)
processor.tokenizer.padding_side = "right"

# --- 2. 挂载 LoRA 权重 ---
print(f"Loading LoRA weights from: {PEFT_MODEL_ID}")
SPECIAL_TOKENS = ["[ANS]", "[REASON]"]
added_tokens = processor.tokenizer.add_tokens(SPECIAL_TOKENS)
if added_tokens and added_tokens > 0:
    base_model.resize_token_embeddings(len(processor.tokenizer))

model = PeftModel.from_pretrained(base_model, PEFT_MODEL_ID)
model = model.merge_and_unload() # 合并权重
model.eval()
model.to(DEVICE)
print("Model loaded and LoRA weights merged.")

# --- 3. 构造 Qwen Chat Prompt ---
# Qwen 的对话格式
messages = [
    {
        "role": "user",
        "content": [
            # 文本问题，但 Qwen-VL 格式需要 content 是列表
            {"type": "text", "text": QUESTION}
        ]
    }
]

# 转换为 Qwen chat template 文本
text_prompt = processor.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True # 必须添加，表示期待助手回答
)

# 纯文本问题，视觉信息为 None
image_inputs, video_inputs = process_vision_info(messages)

print("\n--- Model Input Prompt ---")
print(text_prompt)
print("--------------------------")

# --- 4. 编码并生成 ---
inputs = processor(
    text=[text_prompt],  # 必须是列表
    images=image_inputs,
    videos=video_inputs,
    return_tensors="pt",
).to(DEVICE)

with torch.no_grad():
    # 使用 generate 生成回答
    generated_ids = model.generate(
        **inputs,
        max_new_tokens=128,  # 设置最大生成长度
        do_sample=False,
        eos_token_id=processor.tokenizer.eos_token_id
    )

    # 解码生成的 Token ID
    decoded_text = processor.tokenizer.batch_decode(
        generated_ids[:, inputs.input_ids.shape[1]:],
        skip_special_tokens=True
    )[0]

# --- 5. 打印结果 ---
print("\n--- Model Response ---")
print(f"Q: {QUESTION}")
print(f"A: {decoded_text.strip()}")
print("----------------------")