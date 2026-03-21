import os
import sys
import argparse
import torch
from torch.utils.data import DataLoader

__package__ = "trainer"
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from model.MokioModel import MokioMindConfig
from dataset.lm_dataset import PretrainDataset
from trainer_utils import init_model, Logger, is_main_process


def evaluate(model, tokenizer, eval_loader, device, args):
    """评估模型在验证集上的困惑度"""
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    
    with torch.no_grad():
        for batch in eval_loader:
            # PretrainDataset.__getitem__ 返回 tuple (input_ids, labels, attention_mask)
            input_ids, labels, attention_mask = batch
            input_ids = input_ids.to(device)
            labels = labels.to(device)
            attention_mask = attention_mask.to(device)
            
            # 前向传播
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )
            
            # 计算有效token数量（排除padding和-100的label）
            valid_tokens = (labels != -100).sum().item()
            total_tokens += valid_tokens
            
            # 累加loss（已经处理过shift）
            if outputs.loss is not None:
                total_loss += outputs.loss.item() * valid_tokens
    
    # 计算平均困惑度
    avg_loss = total_loss / total_tokens if total_tokens > 0 else float('inf')
    perplexity = torch.exp(torch.tensor(avg_loss)).item()
    
    model.train()
    return avg_loss, perplexity


def eval_epoch(epoch, loader, model, device, args):
    """评估一个epoch"""
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    step = 0
    
    with torch.no_grad():
        for batch in loader:
            # PretrainDataset.__getitem__ 返回 tuple (input_ids, labels, attention_mask)
            input_ids, labels, attention_mask = batch
            input_ids = input_ids.to(device)
            labels = labels.to(device)
            attention_mask = attention_mask.to(device)
            
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )
            
            # 计算有效token数量
            valid_tokens = (labels != -100).sum().item()
            total_tokens += valid_tokens
            
            if outputs.loss is not None:
                total_loss += outputs.loss.item() * valid_tokens
            
            step += 1
            if step % args.log_interval == 0:
                Logger(f"Eval Step [{step}/{len(loader)}]")
    
    # 计算平均困惑度
    avg_loss = total_loss / total_tokens if total_tokens > 0 else float('inf')
    perplexity = torch.exp(torch.tensor(avg_loss)).item()
    
    return avg_loss, perplexity


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MokioMind Evaluation")
    
    # ========== 评估参数 ==========
    parser.add_argument("--weight_path", type=str, required=True, help="模型权重路径")
    parser.add_argument("--data_path", type=str, default="../dataset/pretrain_hq.jsonl", help="评估数据路径")
    parser.add_argument("--batch_size", type=int, default=32, help="batch size")
    parser.add_argument("--max_seq_len", type=int, default=512, help="最大序列长度")
    parser.add_argument("--num_workers", type=int, default=1, help="数据加载线程数")
    parser.add_argument("--log_interval", type=int, default=10, help="日志打印间隔")
    
    # ========== 模型参数 ==========
    parser.add_argument("--hidden_size", default=512, type=int, help="隐藏层维度")
    parser.add_argument("--num_hidden_layers", default=8, type=int, help="隐藏层数量")
    parser.add_argument("--use_moe", default=0, type=int, choices=[0, 1], help="是否使用MoE架构")
    
    # ========== 硬件参数 ==========
    parser.add_argument(
        "--device",
        type=str,
        default="cuda:0" if torch.cuda.is_available() else "cpu",
        help="评估设备",
    )
    
    args = parser.parse_args()
    
    # 创建模型配置
    lm_config = MokioMindConfig(
        hidden_size=args.hidden_size,
        num_hidden_layers=args.num_hidden_layers,
        use_moe=bool(args.use_moe),
    )
    
    # 加载模型和tokenizer
    Logger("正在加载模型和tokenizer...")
    model, tokenizer = init_model(
        lm_config,
        from_weight="none",  # 不从权重加载，单独加载
        device=args.device
    )
    
    # 加载权重
    if os.path.exists(args.weight_path):
        Logger(f"正在从 {args.weight_path} 加载权重...")
        weights = torch.load(args.weight_path, map_location=args.device)
        model.load_state_dict(weights, strict=False)
        Logger("权重加载成功！")
    else:
        Logger(f"错误：权重文件不存在: {args.weight_path}")
        sys.exit(1)
    
    # 加载评估数据
    Logger(f"正在加载评估数据: {args.data_path}")
    eval_ds = PretrainDataset(args.data_path, tokenizer, max_length=args.max_seq_len)
    eval_loader = DataLoader(
        eval_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    Logger(f"评估数据加载完成，共 {len(eval_ds)} 条样本")
    
    # 开始评估
    Logger("=" * 50)
    Logger("开始评估...")
    Logger("=" * 50)
    
    avg_loss, perplexity = eval_epoch(0, eval_loader, model, args.device, args)
    
    Logger("=" * 50)
    Logger(f"评估完成！")
    Logger(f"Average Loss: {avg_loss:.6f}")
    Logger(f"Perplexity: {perplexity:.6f}")
    Logger("=" * 50)
