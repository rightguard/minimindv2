import os
import sys
import re
import gc
import argparse
import warnings
import torch
import torch.distributed as dist
from torch import optim
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, DistributedSampler
from torch.optim.lr_scheduler import CosineAnnealingLR
from transformers import AutoTokenizer, AutoModel

__package__ = "trainer"
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from model.MokioModel import MokioMindConfig
from dataset.lm_dataset import RLAIFDataset
from trainer.trainer_utils import (
    Logger,
    is_main_process,
    lm_checkpoint,
    init_distributed_mode,
    setup_seed,
    SkipBatchSampler,
    init_model,
)

warnings.filterwarnings("ignore")


def calculate_rewards(prompts, responses, reward_model, reward_tokenizer):
    def reasoning_model_reward(rewards_tensor):
        # 定义正则：pattern 和 pattern2 定义了完美的推理回复格式：
        # 必须以 <think>\n 开头，中间是思考过程，以 </think>\n 结束，紧接着（或者空一行后）是 <answer>\n，最后以 </answer> 结尾。
        pattern = r"^<think>\n.*?\n</think>\n<answer>\n.*?\n</answer>$"
        pattern2 = r"^<think>\n.*?\n</think>\n\n<answer>\n.*?\n</answer>$"

        format_rewards = []
        for response in responses:
            matched = re.match(pattern, response, re.S) or re.match(pattern2, response, re.S)
            format_rewards.append(0.5 if matched else 0.0)
        rewards_tensor += torch.tensor(format_rewards, device=args.device)
    
        def mark_num(text):
            # 功能：定义另一个嵌套函数 mark_num，用来计算“标签数量奖励”。
            # 只要文本里包含且仅包含一个对应的标签，就给 0.25 分。四个标签全对满分是 1.0 分。
            # 这是为了鼓励模型即使整体格式没对，只要输出了正确的标签也能得到部分奖励。
            reward = 0.0
            if text.count("<think>") == 1:
                    reward += 0.25
            if text.count("</think>") == 1:
                    reward += 0.25
            if text.count("<answer>") == 1:
                    reward += 0.25
            if text.count("</answer>") == 1:
                    reward += 0.25
            return reward
    
        rewards_tensor += torch.tensor([mark_num(response) for response in responses], device=args.device)

        return rewards_tensor
    
    # 功能：初始化一个全零的 rewards 张量，长度等于生成的回复数量。
    # 如果系统参数开启了推理模式（args.reasoning == 1），则调用上面定义的格式奖励函数。
    rewards = torch.zeros(len(responses), device=args.device)
    if args.reasoning == 1:
        rewards = reasoning_model_reward(rewards)
    
    with torch.no_grad():
        # 初始化一个列表 reward_model_scores 准备存RM的分数
        # 并设定分数的截断范围边界 scale = 3.0（即分数会被限制在 -3.0 到 3.0 之间）。
        reward_model_scores = []
        scale = 3.0

        # 遍历传入的 prompt。这段代码使用正则表达式解析类似 ChatML 的格式（<|im_start|>role...<|im_end|>）
        # 将其转换成标准的对话列表字典格式（[{"role": "user", "content": "..."}]），这是很多模型推理的标准输入格式。
        for i, prompt in enumerate(prompts):
            pattern = r"<\|im_start\|>(system|user|assistant)\s+(.*?)<\|im_end\|>"
            matches = re.findall(pattern, prompt, re.DOTALL)
            messages = [
                {
                    "role": role,
                    "content": content.strip()
                } for role, content in matches
            ]
            
            # 因为在强化学习（如PPO）中，一个 prompt 往往会生成多个回复（由 args.num_generations 决定）
            # 这里通过数学计算 response_idx 找到当前 prompt 对应的那几个生成回复。
            # 将历史对话 messages 和当前生成的回复拼接成完整的对话 tmp_chat。
            # 调用奖励模型打分，并将分数裁切（clip）到 [-3.0, 3.0] 的范围内。
            for j in range(args.num_generations):
                response_idx = i * args.num_generations + j
                response = responses[response_idx]
                tmp_chat = messages + [{"role": "assistant", "content": response}]
                score = reward_model.get_score(reward_tokenizer, tmp_chat)
                score = max(min(score, scale), -scale)

                # 如果是推理模式，代码还会单独提取 <answer> 标签里面的纯回答内容（忽略 <think> 过程）。
                # 然后让奖励模型只对这个最终回答再打一次分 answer_score（同样裁切到 -3.0 到 3.0）。
                # 最终融合：将整个回复的得分（包含思考过程）和纯回答的得分进行加权平均（整体占 40%，纯回答占 60%）。
                # 这样可以确保模型不仅思考过程好，最终给出的结论也要好。
                if args.reasoning == 1:
                    answer_match = re.search(r"<answer>(.*?)</answer>", response, re.DOTALL)

                    if answer_match:
                        answer_content = answer_match.group(1).strip()
                        answer_chat = messages + [{"role": "assistant", "content": answer_content}]
                        answer_score = reward_model.get_score(reward_tokenizer, answer_chat)
                        answer_score = max(min(answer_score, scale), -scale)
                        score = score * 0.4 + answer_score * 0.6
                reward_model_scores.append(score)
        rewards += torch.tensor(reward_model_scores, device=args.device)
    
    return rewards


def grpo_train_epoch(
    epoch,
    loader,
    iters,
    ref_model,
    reward_model,
    reward_tokenizer,
    start_step=0,
    wandb=None,
):
    for step, batch in enumerate(loader, start=start_step + 1):
        prompts = batch["prompt"]

        prompt_inputs = tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            return_token_type_ids=False,
            add_special_tokens=False,
        ).to(args.device)

        if args.max_seq_len:
            # 段代码的作用是对输入序列进行“左截断”（Left Truncation）
            # 即：如果序列长度超过了预设的最大长度（max_seq_len），则**保留最右侧（最新/最后面）**的词，扔掉左侧较旧的内容。
            # 在 NLP（自然语言处理）任务中，这种写法非常常见，主要原因有以下几点：

            # 1. 语法解释：[:, -args.max_seq_len :] 在干什么？
            # 这是一段典型的 PyTorch/NumPy 切片语法：

            # 第一个 :：表示保留 Batch（批次）维度的所有数据。
            # -args.max_seq_len :：表示从倒数第 max_seq_len 个位置开始，一直取到最后。

            # 举个例子：
            # 假设 max_seq_len = 3，而你的 input_ids 是 [1, 2, 3, 4, 5]。
            # 经过这个操作，它会变成 [3, 4, 5]。

            # 2. 为什么要“舍左保右”？（核心原因）
            # A. 保留最关键的指令信息
            # 在对话模型（如 GPT、Llama、DeepSeek）中，最重要的信息通常在** Prompt 的末尾**。
            # 开头通常是：系统提示词（System Prompt）、久远的对话历史。
            # 结尾通常是：用户当前的提问、刚刚给出的推理指令。
            # 如果因为长度超限要删掉一部分，删掉很久以前的聊天记录显然比删掉当前的提问要合理得多。

            # B. 硬件显存限制 (OOM)
            # Transformer 模型的计算复杂度随长度呈平方级增长。
            # 如果不设置 max_seq_len，一旦输入极长的文本，显存会瞬间炸裂（Out of Memory）。这种写法是一种“强行保命”的防御性编程。

            # C. 模型位置编码的限制
            # 每个模型在训练时都有一个最大上下文窗口（比如 4096 或 8192）。
            # 如果输入 5000 个词给一个只有 4096 窗口的模型，模型无法处理多出来的部分。通过截断，可以确保输入序列始终在模型的“认知范围”内。

            # 3. 为什么不直接用 tokenizer(truncation=True)？
            # 虽然分词器（Tokenizer）自带截断功能，但在强化学习（RLHF）或复杂的推理流水线中，手动切片有以下优势：
            # 灵活性：有时候我们是在拼接了多个字段（比如 System + User + Context）后才发现超长了，手动切片可以更精准地控制。
            # 确保 Tensor 形状一致：在处理 Batch 数据时，手动切片可以确保这一组数据送入模型前，宽度是整齐划一的。
            # 后处理方便：在某些推理场景下，我们需要先拿完整的 sequence 做某些计算，最后才切掉给模型推理。
            prompt_inputs["input_ids"] = prompt_inputs["input_ids"][:, -args.max_seq_len :]
            prompt_inputs["attention_mask"] = prompt_inputs["attention_mask"][:, -args.max_seq_len :]

        with torch.no_grad():
            model_for_gen = (model.module if isinstance(model, DistributedDataParallel) else model)

            # 在模型生成（Inference/Generation）过程中，temperature（温度）是一个非常核心的参数，它直接控制了模型生成内容的**“随机性”或“创造力”**。
            # 简单来说，它就像是一个**“混沌指数”**调节旋钮。1. 直观理解：它在干什么？大语言模型在预测下一个词时，并不是直接给出一个词，而是给所有可能的词计算一个概率得分（Logits）。
            # 低温度 (T < 1.0)：模型变得“保守”和“确定”。
            # 模型会倾向于反复选择概率最高的那个词。输出结果非常稳定、逻辑严密，但可能会陷入死循环或显得生硬（缺乏灵气）。
            # 
            # 高温度 (T > 1.0)：模型变得“冒险”和“狂野”。原本概率较低的词也有了被选中的机会。
            # 输出结果更有创意、多样化，但太高了就会胡言乱语（幻觉严重）。
            # 
            # T = 1.0： 原始概率分布，不做任何干预。
            # T -> 0（趋近于0）： 演变成“贪婪搜索”（Greedy Search），模型永远只选概率最高的那一个词，结果是完全确定的。
            # 
            # 2. 数学原理：Softmax 的变换在深度学习中，模型最后的输出通过 Softmax 函数转化为概率。
            # 引入 temperature (T) 后的公式如下：P_i = {exp(z_i / T) / sum_j{exp(z_j / T)}}
            # 当 T 很大时： z_i / T 的值会变小，原本差距很大的得分被“拉平”了。大家（各个词）的概率变得差不多，模型随机乱选。
            # 当 T 很小时： z_i / T 的值会放大。原本领先一点点的词，经过指数运算后，概率会占据绝对统治地位，压缩了其他词的空间。
            # 
            # 3. 应用场景对比温度设置效果适用场景
            # 0.0 - 0.3：极度精准、确定、重复数学题、代码编写、事实问答、提取摘要
            # 0.5 - 0.8：连贯且有一定灵活性通用对话、文案润色、大部分翻译任务
            # 1.0 - 1.5：极具创意、发散、不可预测写诗、写小说、头脑风暴、角色扮演
            outputs = model_for_gen.generate(
                **prompt_inputs,
                max_new_tokens=args.max_gen_len,
                do_sample=True,
                temperature=0.8,
                num_return_sequences=args.num_generations,
                pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )

        # outputs:
        # 这是模型生成的完整张量，形状通常是 [Batch_Size, Total_Length]。
        # Total_Length = Prompt_Length（提问长度）+ Generated_Length（回答长度）。

        # [:, ...]:
        # 前面的 : 表示保留所有的 Batch（即同时处理这一批次里的所有对话）。

        # prompt_inputs["input_ids"].size(1):
        # 这是你输入给模型的 Prompt 的长度。
        # .size(1) 指的是张量的第二维（序列长度维度）。

        # [ ... : ]:
        # 这是一个切片操作，表示：从“Prompt 长度”这个位置开始，一直取到最后。
        completion_ids = outputs[:, prompt_inputs["input_ids"].size(1) :]

        def get_per_token_logps(mdl, input_ids, n_keep):
            # 假设我们输入了 4 个问题（Prompts），每个问题让模型生成 2 个不同的回答（Num_Generations=2），所以总共有 8 条数据。
            # 假设模型生成的回答长度（Completion）是 20 个 Token，词表大小是 32000。
            # input_ids: 完整的输入（包含问题+回答），形状 [8, 总长度]
            # n_keep: 我们只需要计算"回答"部分的概率，长度是 20

            # # 如果 input_ids 是推理模式生成的（不需要梯度），就克隆一份并切断梯度（detach）
            input_ids = (input_ids.detach().clone() if input_ids.is_inference() else input_ids)

            # 1. 前向传播拿 Logits（得分）
            # logits_to_keep = n_keep + 1，表示只计算最后 21 个词的得分。
            # [:, :-1, :] 表示去掉最后一个词的得分（因为最后一个词没有下一个词可以预测了）
            # logits 最终形状: [8, 20, 32000] -> 8条数据，20个位置，每个位置对32000个词的预测打分
            logits = mdl(input_ids=input_ids, logits_to_keep=n_keep + 1).logits[:, :-1, :]

            # 2. 循环遍历这 8 条数据。
            # zip 把 logits_row (形状 [20, 32000]) 和 ids_row (实际生成的回答，形状 [20]) 绑在一起
            # 每条数据都单独处理：
            # ids_row.detach().clone() 克隆一份（不参与梯度计算）。
            # torch.gather 从 logits_row 中找出 ids_row 对应的概率值。
            # 最终结果：per_token_logps 是 [8, 20]，每个位置对应一个词的 log 概率。
            per_token_logps = []
            for logits_row, ids_row in zip(logits, input_ids[:, -n_keep:]):
                ids_row = (ids_row.detach().clone() if ids_row.is_inference() else ids_row)

                # 3. 核心切片/提取操作：gather
                # logits_row.log_softmax(dim=-1) 把 32000 个得分变成对数概率 (更稳定的概率表示)
                # ids_row.unsqueeze(1) 把 [20] 变成 [20, 1]，为了配合 gather 查表
                # gather 的作用是：在 32000 个概率中，准确地把你"实际生成的那个词"的概率抠出来！
                # squeeze(1) 把抠出来的结果从 [20, 1] 变回 [20]
                token_logps = torch.gather(logits_row.log_softmax(dim=-1), 1, ids_row.unsqueeze(1)).squeeze(1)
                per_token_logps.append(token_logps)

            # 把 8 个 [20] 堆叠起来，返回形状 [8, 20]
            return torch.stack(per_token_logps)

        # 调用上面写的函数，计算当前训练模型 (model) 对生成的 20 个词的概率
        # completion_ids.size(1) 就是 20
        per_token_logps = get_per_token_logps(model, outputs, completion_ids.size(1))

        # ref_model 是"参考模型"（训练前的大模型，权重被冻结了）。
        # 我们要看看"老模型"当初对这句回答的概率是多少，用来限制"新模型"别跑偏。
        with torch.no_grad():
            ref_per_token_logps = get_per_token_logps(ref_model, outputs, completion_ids.size(1))

        # 把 Token ID 解码成人类能看懂的文字文本
        completions = tokenizer.batch_decode(completion_ids, skip_special_tokens=True)

        # 调用你之前发过的计算奖励的代码，拿到 8 个分数。rewards 形状: [8]
        rewards = calculate_rewards(prompts, completions, reward_model, reward_tokenizer).to(args.device)

        # grouped_rewards 把 8 个分数重新排版。
        # 变成 4 行 2 列，每一行代表同一个问题的 2 个不同回答的得分。形状: [4, 2]
        grouped_rewards = rewards.view(-1, args.num_generations)

        # 算同一个问题 2 个回答的平均分 (mean_r) 和标准差 (std_r)
        # repeat_interleave 把算出来的平均分复制展开，变回 [8]
        mean_r = grouped_rewards.mean(dim=1).repeat_interleave(args.num_generations)
        std_r = grouped_rewards.std(dim=1).repeat_interleave(args.num_generations)

        # 算 Advantage (优势)：(当前得分 - 平均分) / 标准差
        # 如果你比平均分高，优势就是正的；比平均分低，就是负的。
        # torch.clamp 限制优势值在 [-10, 10] 之间，防止极端分数把模型带飞
        advantages = torch.clamp((rewards - mean_r) / (std_r + 1e-4), -10, 10)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # is_eos: 找出哪些位置是结束符。形状 [8, 20]，是一个 True/False 的布尔矩阵
        is_eos = completion_ids == tokenizer.eos_token_id

        # eos_idx: 初始化一个全是 20 的一维数组。形状 [8]
        eos_idx = torch.full((is_eos.size(0),), is_eos.size(1), dtype=torch.long, device=args.device)

        # is_eos.any(dim=1) 找出哪些句子确实包含了 EOS
        # is_eos.int().argmax(dim=1) 找出每一行第一个 EOS 出现的位置索引（比如 15）
        eos_idx[is_eos.any(dim=1)] = is_eos.int().argmax(dim=1)[is_eos.any(dim=1)]

        # 极其巧妙的一行代码！生成 Mask 遮罩
        # torch.arange 产生 [0, 1, 2... 19]，形状变 [8, 20]
        # 判断当前位置的数字是不是 <= eos_idx 里的位置
        # 比如 EOS 在 15，那么 0~15 都是 True (1)，16~19 都是 False (0)
        # completion_mask 形状 [8, 20]，全是 1 和 0。
        completion_mask = (torch.arange(is_eos.size(1), device=args.device).expand(is_eos.size(0), -1)<= eos_idx.unsqueeze(1)).int()

        # 计算新老模型的概率差异 (KL 散度)。
        kl_div = ref_per_token_logps - per_token_logps

        # 这里用了一个近似的 KL 散度公式（Schulman 提出的），确保 KL 散度总是正的。形状 [8, 20]
        per_token_kl = torch.exp(kl_div) - kl_div - 1

        # 计算强化学习的策略损失 (Policy Loss)
        per_token_loss = -(
            # torch.exp(新 - 截断新) 是一个数学黑魔法。
            # 它在正向计算时值永远是 1.0，但反向传播求导时恰好等于梯度的方向。
            torch.exp(per_token_logps - per_token_logps.detach())

            # 乘以优势 (把 [8] 变成 [8, 1] 以便和 [8, 20] 广播相乘)。优势越高，这边的梯度推力越大。
            * advantages.unsqueeze(1)

            # 减去 KL 惩罚项。如果模型偏离老模型太多，这里会拉它一把。args.beta 是惩罚力度。
            - args.beta * per_token_kl
        )

        # per_token_loss 的形状是 [8, 20]。
        # 乘以 completion_mask，把 Padding 位置的损失强行变成 0（不计算那些废话）
        # sum(dim=1) 把 20 个词的损失加起来
        # / completion_mask.sum(dim=1) 除以实际的词数，求平均。
        # 最后外面再套一个 .mean()，把 8 条数据的总损失再求平均，得到最终的一个标量数字。
        # 注意：loss 不再除以 accumulation_steps，由 optimizer step 前统一处理
        loss = ((per_token_loss * completion_mask).sum(dim=1) / completion_mask.sum(dim=1)).mean()

        # NaN/inf 检查：防止梯度爆炸或数值不稳定导致训练持续崩溃
        if torch.isnan(loss) or torch.isinf(loss):
            optimizer.zero_grad()
            if step % args.log_interval == 0 and is_main_process():
                Logger(f"Step {step}: loss={loss.item():.6f} (NaN/inf detected, skipped)")
            continue

        loss.backward()

        if step % args.accumulation_steps == 0:
            if args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

        if step % args.log_interval == 0 or step == iters:
            policy_loss_val = loss.item()
            avg_reward_val = rewards.mean().item()
            avg_len_val = completion_mask.sum(dim=1).float().mean().item()
            current_lr = optimizer.param_groups[0]["lr"]

            Logger(
                f"Epoch: {epoch + 1}, Step: {step}/{iters}, "
                f"Actor Loss: {policy_loss_val:.6f}, Reward: {avg_reward_val:.6f}, "
                f"Avg Response Len: {avg_len_val:.2f}, LR: {current_lr:.2e}"
            )

            if wandb and is_main_process():
                wandb.log(
                    {
                        "policy_loss": policy_loss_val,
                        "reward": avg_reward_val,
                        "avg_response_len": avg_len_val,
                        "advantages_mean": advantages.mean().item(),
                        "learning_rate": current_lr,
                    }
                )

        if (step % args.save_interval == 0 or step == iters) and is_main_process():
            model.eval()
            moe_suffix = "_moe" if lm_config.use_moe else ""
            ckp = f"{args.save_dir}/{args.save_weight}_{lm_config.hidden_size}{moe_suffix}.pth"
            state_dict = (model.module.state_dict() if isinstance(model, DistributedDataParallel) else model.state_dict())
            torch.save({k: v.half() for k, v in state_dict.items()}, ckp)
            lm_checkpoint(
                lm_config,
                weight=args.save_weight,
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                step=step,
                wandb=wandb,
                save_dir="../checkpoints",
                scheduler=scheduler,
            )
            model.train()

        del prompt_inputs, outputs, completion_ids, per_token_logps, ref_per_token_logps
        del (
            completions,
            rewards,
            grouped_rewards,
            mean_r,
            std_r,
            advantages,
            completion_mask,
        )
        torch.cuda.empty_cache()
        gc.collect()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="MokioMind GRPO (Group Relative Policy Optimization)"
    )

    parser.add_argument("--save_dir", type=str, default="../out", help="模型保存目录")
    parser.add_argument(
        "--save_weight", default="grpo", type=str, help="保存权重的前缀名"
    )
    parser.add_argument("--epochs", type=int, default=1, help="训练轮数")
    parser.add_argument("--batch_size", type=int, default=2, help="batch size")
    parser.add_argument("--learning_rate", type=float, default=8e-8, help="初始学习率")

    parser.add_argument(
        "--device",
        type=str,
        default="cuda:0" if torch.cuda.is_available() else "cpu",
        help="训练设备",
    )
    parser.add_argument("--dtype", type=str, default="bfloat16", help="混合精度类型")
    parser.add_argument("--num_workers", type=int, default=1, help="数据加载线程数")

    parser.add_argument(
        "--accumulation_steps", type=int, default=1, help="梯度累积步数"
    )
    parser.add_argument("--grad_clip", type=float, default=1.0, help="梯度裁剪阈值")
    parser.add_argument("--log_interval", type=int, default=1, help="日志打印间隔")
    parser.add_argument("--save_interval", type=int, default=10, help="模型保存间隔")

    parser.add_argument("--hidden_size", default=512, type=int, help="隐藏层维度")
    parser.add_argument("--num_hidden_layers", default=8, type=int, help="隐藏层数量")
    parser.add_argument(
        "--use_moe",
        default=0,
        type=int,
        choices=[0, 1],
        help="是否使用MoE架构（0=否，1=是）",
    )

    parser.add_argument("--max_seq_len", default=66, type=int, help="Prompt最大长度")
    parser.add_argument("--max_gen_len", type=int, default=1536, help="生成的最大长度")

    parser.add_argument(
        "--data_path",
        type=str,
        default="../dataset/rlaif-mini.jsonl",
        help="RLAIF数据路径",
    )
    parser.add_argument(
        "--num_generations", type=int, default=8, help="每个prompt生成的样本数"
    )
    parser.add_argument("--beta", type=float, default=0.02, help="KL惩罚系数")
    parser.add_argument(
        "--reasoning",
        type=int,
        default=1,
        choices=[0, 1],
        help="推理模型类型（0=普通模型，1=推理模型）",
    )
    parser.add_argument(
        "--reward_model_path",
        type=str,
        default="../../internlm2-1_8b-reward",
        help="Reward模型路径",
    )
    parser.add_argument(
        "--from_resume",
        default=0,
        type=int,
        choices=[0, 1],
        help="是否自动检测&续训（0=否，1=是）",
    )

    parser.add_argument("--use_wandb", action="store_true", help="是否使用wandb")
    parser.add_argument(
        "--wandb_project", type=str, default="MokioMind-GRPO", help="wandb项目名"
    )
    args = parser.parse_args()

    local_rank = init_distributed_mode()
    if dist.is_initialized():
        args.device = f"cuda:{local_rank}"
    setup_seed(42 + (dist.get_rank() if dist.is_initialized() else 0))

    os.makedirs(args.save_dir, exist_ok=True)
    lm_config = MokioMindConfig(
        hidden_size=args.hidden_size,
        num_hidden_layers=args.num_hidden_layers,
        max_position_embeddings=args.max_seq_len + args.max_gen_len,
        use_moe=bool(args.use_moe),
    )
    ckp_data = (
        lm_checkpoint(lm_config, weight=args.save_weight, save_dir="../checkpoints")
        if args.from_resume == 1
        else None
    )

    wandb = None
    if args.use_wandb and is_main_process():
        import swanlab as wandb

        wandb_id = ckp_data.get("wandb_id") if ckp_data else None
        resume = "must" if wandb_id else None
        wandb_run_name = f"MokioMind-GRPO-Epoch-{args.epochs}-BS-{args.batch_size}-LR-{args.learning_rate}"
        wandb.init(
            project=args.wandb_project, name=wandb_run_name, id=wandb_id, resume=resume
        )

    base_weight = "reason" if args.reasoning == 1 else "full_sft"

    model, tokenizer = init_model(lm_config, base_weight, device=args.device)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    ref_model, _ = init_model(lm_config, base_weight, device=args.device)
    ref_model = ref_model.eval().requires_grad_(False)

    reward_model = AutoModel.from_pretrained(
        args.reward_model_path, torch_dtype=torch.float16, trust_remote_code=True
    )
    reward_model = reward_model.to(args.device).eval().requires_grad_(False)
    reward_tokenizer = AutoTokenizer.from_pretrained(
        args.reward_model_path, trust_remote_code=True
    )

    train_ds = RLAIFDataset(
        args.data_path, tokenizer, max_length=lm_config.max_position_embeddings
    )
    train_sampler = DistributedSampler(train_ds) if dist.is_initialized() else None
    optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate)

    loader_for_count = DataLoader(
        train_ds, batch_size=args.batch_size, sampler=train_sampler
    )
    iters = len(loader_for_count)
    total_optimizer_steps = max(1, (iters // args.accumulation_steps) * args.epochs)
    scheduler = CosineAnnealingLR(
        optimizer, T_max=total_optimizer_steps, eta_min=args.learning_rate / 10
    )

    start_epoch, start_step = 0, 0
    if ckp_data:
        model.load_state_dict(ckp_data["model"])
        optimizer.load_state_dict(ckp_data["optimizer"])
        scheduler.load_state_dict(ckp_data["scheduler"])
        start_epoch = ckp_data["epoch"]
        start_step = ckp_data.get("step", 0)

    if dist.is_initialized():
        model._ddp_params_and_buffers_to_ignore = {"freqs_cos", "freqs_sin"}
        model = DistributedDataParallel(model, device_ids=[local_rank])

    for epoch in range(start_epoch, args.epochs):
        train_sampler and train_sampler.set_epoch(epoch)

        if epoch == start_epoch and start_step > 0:
            batch_sampler = SkipBatchSampler(
                train_sampler or range(len(train_ds)), args.batch_size, start_step
            )
            loader = DataLoader(
                train_ds,
                batch_sampler=batch_sampler,
                num_workers=args.num_workers,
                pin_memory=True,
            )
            Logger(
                f"Epoch [{epoch + 1}/{args.epochs}]: 跳过前{start_step}个step，从step {start_step + 1}开始"
            )
            grpo_train_epoch(
                epoch,
                loader,
                len(loader) + start_step,
                ref_model,
                reward_model,
                reward_tokenizer,
                start_step,
                wandb,
            )
        else:
            loader = DataLoader(
                train_ds,
                batch_size=args.batch_size,
                pin_memory=True,
                drop_last=False,
                shuffle=(train_sampler is None),
                num_workers=args.num_workers,
                sampler=train_sampler,
            )
            grpo_train_epoch(
                epoch,
                loader,
                len(loader),
                ref_model,
                reward_model,
                reward_tokenizer,
                0,
                wandb,
            )

    
    
            