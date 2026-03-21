import os
import sys
import shutil
import glob
import subprocess

# ==============================================================================
# 1. 路径与流程配置区 (Kaggle 专用)
# ==============================================================================
# 请将这里的路径替换为你实际在 Kaggle Add Data 后得到的绝对路径
DATA_PRETRAIN = "/kaggle/input/datasets/rightguard/minimind-v2data/pretrain_hq.jsonl"
DATA_SFT      = "/kaggle/input/datasets/rightguard/minimind-v2data/sft_mini_512.jsonl"
DATA_LORA     = "/kaggle/input/datasets/rightguard/minimind-v2data/lora_medical.jsonl"
DATA_DPO      = "/kaggle/input/datasets/rightguard/minimind-v2data/dpo.jsonl"
DATA_GRPO = None

# 奖励模型：直接填入 HuggingFace 的模型 ID，代码会自动联网下载！无需手动上传！
REWARD_MODEL_PATH = "internlm/internlm2-1_8b-reward"

# 流程开关：如果你不想跑某一步，把它改成 False 即可跳过
RUN_PRETRAIN = True
RUN_SFT      = True
RUN_LORA     = True
RUN_DPO      = True
RUN_GRPO     = False  # ⚠️ 默认关闭：因为奖励模型很大，建议前4步跑通后，单独开 True 跑测试

# 输出与权重目录 (Kaggle 建议保存在 working 目录下)
OUT_DIR = "../out"
WEIGHT_DIR = "../weight"
LORA_WEIGHT_DIR = os.path.join(WEIGHT_DIR, "lora")
CHECKPOINT_DIR = "../checkpoints"

# 显卡数量 (Kaggle T4x2 请保持为 2，P100 请改为 1)
NUM_GPUS = 1

# ==============================================================================
# 2. 核心调度逻辑
# ==============================================================================
def setup_directories():
    print("📁 正在创建必要的输出目录...")
    for d in [OUT_DIR, WEIGHT_DIR, LORA_WEIGHT_DIR, CHECKPOINT_DIR]:
        os.makedirs(d, exist_ok=True)

def run_step(step_name, cmd_args):
    """执行单个训练步骤并捕获报错"""
    if NUM_GPUS > 1:
        base_cmd = ["torchrun", f"--nproc_per_node={NUM_GPUS}"]
    else:
        base_cmd = ["python"]
    
    full_cmd = base_cmd + [str(x) for x in cmd_args]
    cmd_str = " ".join(full_cmd)
    
    print("=" * 60)
    print(f"▶ 开始执行: {step_name}")
    print(f"💻 运行命令: {cmd_str}")
    
    result = subprocess.run(full_cmd)
    
    if result.returncode != 0:
        print(f"❌ {step_name} 失败，进程意外退出！状态码: {result.returncode}")
        sys.exit(1)
    print(f"✅ {step_name} 执行成功！\n")

def copy_final_weights(prefix, source_dir=OUT_DIR, target_dir=WEIGHT_DIR):
    """将训练好的模型权重拷贝到统一的 weight 文件夹"""
    pattern = os.path.join(source_dir, f"{prefix}_*.pth")
    files = glob.glob(pattern)
    for f in files:
        shutil.copy(f, target_dir)
        print(f"📦 已将权重拷贝至: {target_dir}/{os.path.basename(f)}")

# ==============================================================================
# 3. 流水线执行
# ==============================================================================
if __name__ == "__main__":
    print("🚀 启动 MokioMind 全流程自动化训练流水线...")
    setup_directories()

    # --- 阶段 1: Pretrain ---
    if RUN_PRETRAIN:
        run_step("1/5 Pretrain 预训练", [
            "train_pretrain.py",
            "--data_path", DATA_PRETRAIN,
            "--save_dir", OUT_DIR,
            "--weight_dir", WEIGHT_DIR,
            "--save_weight", "pretrain",
            "--epochs", 1,
            "--batch_size", 16,
            "--learning_rate", 5e-4
        ])

    # --- 阶段 2: Full SFT ---
    if RUN_SFT:
        run_step("2/5 Full SFT 全量微调", [
            "train_full_sft.py",
            "--data_path", DATA_SFT,
            "--from_weight", "pretrain",
            "--save_dir", OUT_DIR,
            "--save_weight", "full_sft",
            "--epochs", 2,
            "--batch_size", 16,
            "--learning_rate", 1e-5
        ])
        copy_final_weights("full_sft")

    # --- 阶段 3: LoRA ---
    if RUN_LORA:
        run_step("3/5 LoRA 专项微调", [
            "train_lora.py",
            "--data_path", DATA_LORA,
            "--from_weight", "full_sft",
            "--save_dir", LORA_WEIGHT_DIR,
            "--lora_name", "lora_medical",
            "--epochs", 5,
            "--batch_size", 32,
            "--learning_rate", 1e-4
        ])

    # --- 阶段 4: DPO ---
    if RUN_DPO:
        run_step("4/5 DPO 偏好对齐", [
            "train_dpo.py",
            "--data_path", DATA_DPO,
            "--from_weight", "full_sft",
            "--save_dir", OUT_DIR,
            "--save_weight", "dpo",
            "--epochs", 1,
            "--batch_size", 4,
            "--learning_rate", 5e-6
        ])
        copy_final_weights("dpo")

    # --- 阶段 5: GRPO ---
    if RUN_GRPO:
        run_step("5/5 GRPO 强化学习", [
            "train_grpo.py",
            "--data_path", DATA_GRPO,
            "--reward_model_path", REWARD_MODEL_PATH,  # 使用自动下载的模型 ID
            "--reasoning", 1,
            "--save_dir", OUT_DIR,
            "--save_weight", "grpo",
            "--epochs", 1,
            "--batch_size", 1, 
            "--learning_rate", 1e-6
        ])
        copy_final_weights("grpo")

    print("\n🎉 恭喜！所选训练流程全部成功完成！")
    print(f"👉 最终所有模型权重均已汇总至目录: {os.path.abspath(WEIGHT_DIR)}")