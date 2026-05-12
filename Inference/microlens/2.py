from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="meta-llama/Meta-Llama-3-8B-Instruct",
    local_dir="./Meta-Llama-3-8B-Instruct",
    token="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"
)
# snapshot_download(
#     repo_id="meta-llama/Meta-Llama-3-8B-Instruct",
#     local_dir="./Meta-Llama-3-8B-Instruct",
#     token="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",
#     resume_download=True,  # 支持续传
#     max_workers=2          # 限制并发，减少连接压力
# )