import os
import sys

# 📚 Python模块系统
__package__ = "trainer"
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import argparse  # 命令行参数解析
import time  # 时间统计
import torch  # PyTorch深度学习框架
import torch.nn.functional as F  # 神经网络函数
import torch.distributed as dist  # 分布式训练支持
from contextlib import nullcontext  # 上下文管理器
from torch import optim  # 优化器
from torch.nn.parallel import DistributedDataParallel  # 分布式数据并行
from torch.utils.data import DataLoader, DistributedSampler  # 数据加载

# MokioMind相关组件
from model.MokioModel import MokioMindConfig  # 模型配置
from dataset.lm_dataset import DPODataset  # DPO数据集
from trainer.trainer_utils import (  # 训练工具函数
    get_lr,
    Logger,
    is_main_process,
    lm_checkpoint,
    init_distributed_mode,
    setup_seed,
    init_model,
    SkipBatchSampler,
)

def logits_to_log_probs(logits, labels):
    # 词表logits转换为log概率
    # logits: [batch_size, seq_len, vocab_size]
    # labels: [batch_size, seq_len]
    log_probs = F.log_softmax(logits, dim=2)


    # 从log词表概率里选出label对应的log概率
    # 也就是从拿到token在其对应位置的概率
    '''
    torch.gather 的作用是按索引取值。
    在 LLM 训练中，它可以被形象地理解为：“从词典里查出正确答案的得分”。
    1. 为什么要用 unsqueeze(2)?log_probs 是 3D 的 [B, L, V]，而 labels 是 2D 的 [B, L]。
    为了让它们能对齐，我们必须给 labels 增加一个维度：labels.unsqueeze(2) → 形状变为 [B, L, 1]。
    现在，它在第三个维度（深度）上只有一个值，这个值就是我们要捞取的“索引”。
    
    2. gather 的执行过程
    Python
    log_probs_per_token = torch.gather(log_probs, dim=2, index=labels.unsqueeze(2))
    这行代码的逻辑是：锁定坐标：遍历 log_probs 的每一个 Batch (B) 和每一个位置 (L)。
    按图索骥：看 labels 对应位置写的是几（比如是 123）。
    提取数值：去 log_probs 的第 B 行、第 L 列中，把下标为 123 的那个对数概率值（Logit）给抠出来。
    
    3. 一个直观的小例子假设我们只有一个样本 (B=1)，长度为 2 (L=2)，词表大小为 3 (V=3)。
    log_probs (3D):[[ [-0.1, -2.5, -4.0],  # 位置0：模型觉得Token 0概率最大
                    [-3.0, -0.2, -1.5] ] ]]   # 位置1：模型觉得Token 1概率最大
    
    
    labels (2D):[[ 0, 2 ]]  # 正确答案分别是：位置0是Token 0，位置1是Token 2
    gather 之后:它会从第一行取索引 0 的值 -0.1，从第二行取索引 2 的值 -1.5。最后结果（Squeeze 后）：[-0.1, -1.5]。
    '''
    log_probs_per_token = torch.gather(log_probs, dim=2, index=labels.unsqueeze(2)).squeeze(-1)
    return log_probs_per_token

# DPO的loss计算
# 公式：L = -log(σ(β * (π(y_w) - π(y_l) - (π_ref(y_w) - π_ref(y_l)))))
def dpo_loss(ref_log_probs, policy_log_probs, mask, beta):
    seq_lengths = mask.sum(dim=1, keepdim=True).clamp_min(1e-8)  # ！修正：原clamp_min断裂为独立一行，导致NameError
    # 计算ref和policy的序列log概率均值

    ref_log_probs = (ref_log_probs * mask).sum(dim=1) / seq_lengths.squeeze()
    policy_log_probs = (policy_log_probs * mask).sum(dim=1) / seq_lengths.squeeze()

    # 分别获取chosen和rejected的ref和policy的log概率
    # Policy 和 Ref 是什么？
    # 与 PPO 的对比在 DPO 中，我们同时运行两个模型：
    # Policy Model (pi_theta)： 这是当前正在训练的模型。
    # 目标是学习如何提高 Chosen 的概率并降低 Rejected 的概率。
    # Reference Model (pi_ref)： 这是冻结（Frozen）的模型，通常是训练开始前的 SFT 模型。
    # 作用： 作为一个“标尺”，防止 Policy 模型为了追求高分而跑偏（比如变成复读机）。
    # 它保证了模型在优化的同时，不会离原始的语言能力太远。

    # 为什么 batch_size 要整除 2？
    # 这是因为在 DPO 训练的数据构造阶段，为了方便一次性送入显卡计算，开发者通常会将 Chosen 样本和 Rejected 样本在第 0 维度（Batch 维度）上进行了拼接（Concatenate）。
    # 数据结构： 如果你的原始 Batch Size（Prompt 的数量）是 N，那么拼接后的 ref_log_probs 的长度就是 2N。
    # 排列方式： 通常前半部分（索引 0 ... N-1）全是 Chosen 样本，后半部分（索引 N ... 2N-1）全是与之对应的 Rejected 样本。
    # 逻辑拆解：batch_size // 2 拿到的就是原始的 N。
    # ref_log_probs[: batch_size // 2]：切出前半段，即所有的“好答案”。
    # ref_log_probs[batch_size // 2 :]：切出后半段，即所有的“坏答案”。
    # 这样做的目的是让模型在一次前向传播（Forward Pass）中处理完所有数据，提高并行效率，然后再在 Loss 计算时拆开做对比。
    batch_size = ref_log_probs.shape[0]
    chosen_ref_log_probs = ref_log_probs[: batch_size // 2]
    reject_ref_log_probs = ref_log_probs[batch_size // 2 :]
    chosen_policy_log_probs = policy_log_probs[: batch_size // 2]
    reject_policy_log_probs = policy_log_probs[batch_size // 2 :]

    # 计算策略模型的log概率差异
    pi_logratios = chosen_policy_log_probs - reject_policy_log_probs

    # 参考模型的log概率差异
    ref_logratios = chosen_ref_log_probs - reject_ref_log_probs
    
    # DPO损失计算
    # F.logsigmoid 的数学公式
    # F.logsigmoid(x) 是对 Sigmoid 函数取对数，数学表达如下：
    # logsigmoid(x) = log(sigma(x)) = log(1/(1 + e^{-x}))
    # 如果展开写，它可以转化为：logsigmoid(x) = -log(1 + e^{-x})  
    logits = pi_logratios - ref_logratios
    loss = -F.logsigmoid(beta * logits)
    return loss.mean()




def train_epoch(epoch, loader, iters, ref_model, lm_config, start_step=0, wandb=None, beta=0.1):
    start_time = time.time()
    for step, batch in enumerate(loader, start=start_step + 1):
        x_chosen = batch["x_chosen"].to(args.device)
        x_rejected = batch["x_rejected"].to(args.device)
        y_chosen = batch["y_chosen"].to(args.device)
        y_rejected = batch["y_rejected"].to(args.device)
        mask_chosen = batch["mask_chosen"].to(args.device)
        mask_rejected = batch["mask_rejected"].to(args.device)
        attention_mask_chosen = batch["attention_mask_chosen"].to(args.device)  # ！修正：加入attention_mask
        attention_mask_rejected = batch["attention_mask_rejected"].to(args.device)  # ！修正：加入attention_mask

        x = torch.cat([x_chosen, x_rejected], dim=0)
        y = torch.cat([y_chosen, y_rejected], dim=0)
        mask = torch.cat([mask_chosen, mask_rejected], dim=0)
        attention_mask = torch.cat([attention_mask_chosen, attention_mask_rejected], dim=0)  # ！修正：合并attention_mask

        # 📚 学习率调度
        lr = get_lr(epoch * iters + step, args.epochs * iters, args.learning_rate)
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr

        with autocast_ctx:
            # 📚 参考模型前向传播
            # 参考模型冻结，只用于计算baseline概率
            with torch.no_grad():
                ref_outputs = ref_model(x, attention_mask=attention_mask)  # ！修正：加入attention_mask
                ref_logits = ref_outputs.logits
            ref_log_probs = logits_to_log_probs(ref_logits, y)

            # 📚 策略模型前向传播
            # 策略模型是需要优化的主要模型
            outputs = model(
                x, attention_mask=attention_mask
            )  # ！修正：加入attention_mask
            logits = outputs.logits
            policy_log_probs = logits_to_log_probs(logits, y)

            # 📚 DPO损失计算
            dpo_loss_val = dpo_loss(ref_log_probs, policy_log_probs, mask, beta=beta)
            loss = (
                dpo_loss_val + outputs.aux_loss
            )  # ！修正：原缺少aux_loss，MoE辅助损失被丢弃

        # 📚 反向传播
        # 注意：loss 不再除以 accumulation_steps
        scaler.scale(loss).backward()

        if step % args.accumulation_steps == 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)

        # 📚 训练日志
        if step % args.log_interval == 0 or step == iters:
            spend_time = time.time() - start_time
            current_loss = loss.item()
            current_lr = optimizer.param_groups[-1]["lr"]
            eta_min = spend_time / (step + 1) * iters // 60 - spend_time // 60

            Logger(
                f"Epoch:[{epoch + 1}/{args.epochs}]({step}/{iters}) loss:{current_loss:.6f} lr:{current_lr:.12f} epoch_Time:{eta_min}min:"
            )

            if wandb:
                wandb.log(
                    {"loss": current_loss, "lr": current_lr, "epoch_Time": eta_min}
                )

        # 📚 模型保存
        if (step % args.save_interval == 0 or step == iters) and is_main_process():
            model.eval()
            moe_suffix = "_moe" if lm_config.use_moe else ""
            ckp = f"{args.save_dir}/{args.save_weight}_{lm_config.hidden_size}{moe_suffix}.pth"
            raw_model = (
                model.module if isinstance(model, torch.nn.parallel.DistributedDataParallel) else model
            )
            raw_model = getattr(raw_model, "_orig_mod", raw_model)
            state_dict = raw_model.state_dict()
            state_dict = {k: v.half() for k, v in state_dict.items()}  # 半精度保存
            torch.save(state_dict, ckp)
            lm_checkpoint(
                lm_config,
                weight=args.save_weight,
                model=model,
                optimizer=optimizer,
                scaler=scaler,
                epoch=epoch,
                step=step,
                wandb=wandb,
                save_dir="../checkpoints",
            )
            model.train()


if __name__ == "__main__":
    """
    DPO主函数：直接偏好优化脚本的入口点
    
    📚 DPO训练流程：
    1. 准备策略模型和参考模型
    2. 加载偏好数据（chosen vs rejected）
    3. 同时前向传播计算两种模型的概率
    4. 计算DPO损失并优化策略模型
    5. 迭代直到收敛
    """

    # 📚 命令行参数解析
    parser = argparse.ArgumentParser(
        description="MokioMind DPO (Direct Preference Optimization)"
    )

    # ========== 基础训练参数 ==========
    parser.add_argument("--save_dir", type=str, default="../out", help="模型保存目录")
    parser.add_argument(
        "--save_weight", default="dpo", type=str, help="保存权重的前缀名"
    )
    parser.add_argument(
        "--epochs", type=int, default=1, help="训练轮数（DPO通常1-2轮）"
    )
    parser.add_argument(
        "--batch_size", type=int, default=4, help="batch size（DPO batch较小）"
    )

    # 📚 DPO学习率知识点
    # DPO学习率通常很小，避免过度优化导致遗忘
    # 建议不超过5e-8
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=4e-8,
        help="初始学习率（建议<=5e-8避免遗忘）",
    )

    # ========== 硬件配置 ==========
    parser.add_argument(
        "--device",
        type=str,
        default="cuda:0" if torch.cuda.is_available() else "cpu",
        help="训练设备",
    )
    parser.add_argument("--dtype", type=str, default="bfloat16", help="混合精度类型")
    parser.add_argument("--num_workers", type=int, default=1, help="数据加载线程数")

    # ========== 训练策略 ==========
    parser.add_argument(
        "--accumulation_steps", type=int, default=1, help="梯度累积步数"
    )
    parser.add_argument("--grad_clip", type=float, default=1.0, help="梯度裁剪阈值")
    parser.add_argument("--log_interval", type=int, default=5000, help="日志打印间隔")
    parser.add_argument("--save_interval", type=int, default=5000, help="模型保存间隔")

    # ========== 模型架构参数 ==========
    parser.add_argument("--hidden_size", default=512, type=int, help="隐藏层维度")
    parser.add_argument("--num_hidden_layers", default=8, type=int, help="隐藏层数量")
    parser.add_argument(
        "--max_seq_len", default=1024, type=int, help="训练的最大截断长度"
    )
    parser.add_argument(
        "--use_moe",
        default=0,
        type=int,
        choices=[0, 1],
        help="是否使用MoE架构（0=否，1=是）",
    )

    # ========== DPO数据和模型参数 ==========
    # 📚 DPO数据格式知识点
    # 数据包含chosen（偏好）和rejected（不偏好）回答配对
    parser.add_argument(
        "--data_path", type=str, default="../dataset/dpo.jsonl", help="DPO训练数据路径"
    )

    # 📚 DPO权重继承知识点
    # DPO通常基于SFT模型进行对齐优化
    parser.add_argument(
        "--from_weight",
        default="full_sft",
        type=str,
        help="基于哪个权重训练（通常是SFT模型）",
    )
    parser.add_argument(
        "--from_resume",
        default=0,
        type=int,
        choices=[0, 1],
        help="是否自动检测&续训（0=否，1=是）",
    )

    # 📚 DPO beta参数知识点
    # beta控制优化强度，0.1-0.5是常见范围
    parser.add_argument(
        "--beta", default=0.1, type=float, help="DPO中的beta参数（控制优化强度）"
    )

    # ========== 实验跟踪 ==========
    parser.add_argument("--use_wandb", action="store_true", help="是否使用wandb")
    parser.add_argument(
        "--wandb_project", type=str, default="MokioMind-DPO", help="wandb项目名"
    )

    args = parser.parse_args()

    # ========== 1. 初始化环境和随机种子 ==========
    local_rank = init_distributed_mode()
    if dist.is_initialized():
        args.device = f"cuda:{local_rank}"
    setup_seed(42 + (dist.get_rank() if dist.is_initialized() else 0))

    # ========== 2. 配置目录、模型参数、检查ckp ==========
    os.makedirs(args.save_dir, exist_ok=True)
    lm_config = MokioMindConfig(
        hidden_size=args.hidden_size,
        num_hidden_layers=args.num_hidden_layers,
        use_moe=bool(args.use_moe),
    )
    ckp_data = (
        lm_checkpoint(lm_config, weight=args.save_weight, save_dir="../checkpoints")
        if args.from_resume == 1
        else None
    )

    # ========== 3. 设置混合精度 ==========
    device_type = "cuda" if "cuda" in args.device else "cpu"
    dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float16
    autocast_ctx = (
        nullcontext() if device_type == "cpu" else torch.cuda.amp.autocast(dtype=dtype)
    )

    # ========== 4. 配置wandb ==========
    wandb = None
    if args.use_wandb and is_main_process():
        import swanlab as wandb

        wandb_id = ckp_data.get("wandb_id") if ckp_data else None
        resume = "must" if wandb_id else None
        wandb_run_name = f"MokioMind-DPO-Epoch-{args.epochs}-BatchSize-{args.batch_size}-LR-{args.learning_rate}"
        wandb.init(
            project=args.wandb_project, name=wandb_run_name, id=wandb_id, resume=resume
        )

    # ========== 5. 定义模型和参考模型 ==========
    # 📚 DPO双模型架构
    # 策略模型：需要优化的模型
    # 参考模型：冻结的baseline模型
    model, tokenizer = init_model(lm_config, args.from_weight, device=args.device)
    Logger(
        f"策略模型总参数量：{sum(p.numel() for p in model.parameters()) / 1e6:.3f} M"
    )

    # 📚 参考模型初始化
    # 参考模型与策略模型初始权重相同，但完全冻结
    ref_model, _ = init_model(lm_config, args.from_weight, device=args.device)
    ref_model.eval()  # 设为评估模式
    ref_model.requires_grad_(False)  # 冻结所有参数
    Logger(
        f"参考模型总参数量：{sum(p.numel() for p in ref_model.parameters()) / 1e6:.3f} M"
    )

    # 📚 DPO数据集
    train_ds = DPODataset(args.data_path, tokenizer, max_length=args.max_seq_len)
    train_sampler = DistributedSampler(train_ds) if dist.is_initialized() else None
    scaler = torch.cuda.amp.GradScaler(enabled=(args.dtype in ("float16", "bfloat16")))
    optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate)

    # ========== 6. 从ckp恢复状态 ==========
    start_epoch, start_step = 0, 0
    if ckp_data:
        model.load_state_dict(ckp_data["model"])
        optimizer.load_state_dict(ckp_data["optimizer"])
        scaler.load_state_dict(ckp_data["scaler"])
        start_epoch = ckp_data["epoch"]
        start_step = ckp_data.get("step", 0)

    # ========== 7. DDP包装模型 ==========
    if dist.is_initialized():
        model._ddp_params_and_buffers_to_ignore = {"freqs_cos", "freqs_sin"}
        model = DistributedDataParallel(model, device_ids=[local_rank])

    # ========== 8. 开始训练 ==========
    for epoch in range(start_epoch, args.epochs):
        train_sampler and train_sampler.set_epoch(epoch)
        if epoch == start_epoch and start_step > 0:  # 第一个epoch且存在检查点
            batch_sampler = SkipBatchSampler(
                train_sampler or range(len(train_ds)), args.batch_size, start_step
            )
            loader = DataLoader(
                train_ds,
                batch_sampler=batch_sampler,
                num_workers=args.num_workers,
                pin_memory=True,
                persistent_workers=True if args.num_workers > 0 else False,
            )
            Logger(
                f"Epoch [{epoch + 1}/{args.epochs}]: 跳过前{start_step}个step，从step {start_step + 1}开始"
            )
            train_epoch(
                epoch,
                loader,
                len(loader) + start_step,
                ref_model,
                lm_config,
                start_step,
                wandb,
                args.beta,
            )
        else:  # 默认从头开始
            loader = DataLoader(
                train_ds,
                batch_size=args.batch_size,
                shuffle=(train_sampler is None),
                sampler=train_sampler,
                num_workers=args.num_workers,
                pin_memory=True,
                persistent_workers=True if args.num_workers > 0 else False,
            )
            train_epoch(
                epoch, loader, len(loader), ref_model, lm_config, 0, wandb, args.beta
            )