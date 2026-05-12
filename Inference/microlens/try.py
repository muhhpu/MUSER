from transformers import AutoTokenizer, AutoModelForCausalLM
import torch


model_id = "meta-llama/Meta-Llama-3-8B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_id, token='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')
model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.bfloat16, attn_implementation="flash_attention_2", token='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').eval()
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = 'left'
model.generation_config.pad_token_id = model.generation_config.eos_token_id