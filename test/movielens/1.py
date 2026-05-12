import os
import subprocess
import itertools
import time


GPU_SET = "1,2,5,6,7"

lambda_values = [0.1, 0.01, 0.001, 0.0001]
combinations = list(itertools.product(lambda_values, lambda_values))  # 16 组合

for idx, (l1, l2) in enumerate(combinations):
    save_dir = (
        f"/home/team//MLLM-MSR-main/MLLM-MSR/save/LLaVA/"
        f"movielens-lora-recurrent-user-longshort-finetunereason-kua-linear1-e4-r8-numeric-双惩loss-"
        f"l1_{l1}-l2_{l2}"
    )

    print("\n====================================================")
    print(f"▶️ 启动训练任务 {idx+1}/16")
    print(f"   λ1={l1}, λ2={l2}")
    print(f"   SAVE_DIR={save_dir}")
    print(f"   使用 GPU {GPU_SET}")
    print("====================================================\n")

    cmd = (
        f"CUDA_VISIBLE_DEVICES={GPU_SET} "
        f"python test/movielens/test_with_llava_sft.py "
        f"--save_dir {save_dir}"
    )

    # 顺序执行（阻塞直到该训练结束）
    process = subprocess.Popen(cmd, shell=True)
    process.wait()

    print(f"\n✔️ 训练任务 {idx+1}/16 完成！等待 20 秒后继续下一个任务...\n")
    time.sleep(20)  # 给显卡和系统缓一下

print("\n🎉 所有 16 个 λ1/λ2 组合训练全部完成！！！ 🎉")
