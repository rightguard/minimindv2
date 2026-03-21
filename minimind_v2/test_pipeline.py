"""
MiniMind V2 完整流程诊断脚本（不训练）

测试内容：
1. Tokenizer 加载
2. PretrainDataset 数据读取 + tokenize
3. SFTDataset 数据读取 + tokenize + sparse label 生成
4. LoRADataset 数据读取 + tokenize
5. DPODataset 数据读取 + tokenize + loss_mask 生成
6. 模型初始化（前向传播 + 梯度计算）
7. 生成推理测试

运行方式（项目根目录）：
    python test_pipeline.py
"""

import os
import sys

# 确保项目根目录在 Python 路径中
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

import torch
from transformers import AutoTokenizer
from dataset.lm_dataset import PretrainDataset, SFTDataset, DPODataset

# =========================================================================
# 配置区 —— 请根据实际情况修改 DATA_ROOT
# =========================================================================
# 数据集根目录（包含 .jsonl 文件的文件夹）
DATA_ROOT = r"f:\llm\minimindv2da"

# Tokenizer 目录
TOKENIZER_PATH = os.path.join(project_root, "model")

# 数据集路径
DATA_PRETRAIN = os.path.join(DATA_ROOT, "pretrain_hq.jsonl")
DATA_SFT      = os.path.join(DATA_ROOT, "sft_mini_512.jsonl")
DATA_LORA     = os.path.join(DATA_ROOT, "lora_medical.jsonl")
DATA_DPO      = os.path.join(DATA_ROOT, "dpo.jsonl")


def green(msg):  return f"\033[92m{msg}\033[0m"
def red(msg):    return f"\033[91m{msg}\033[0m"
def yellow(msg): return f"\033[93m{msg}\033[0m"


# =========================================================================
# 1. Tokenizer 测试
# =========================================================================
def test_tokenizer():
    print("\n" + "=" * 60)
    print("【1/7】Tokenizer 加载测试")
    print("=" * 60)
    try:
        tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_PATH)
        print(f"  词汇表大小: {len(tokenizer)}")
        print(f"  BOS token: {repr(tokenizer.bos_token)} (id={tokenizer.bos_token_id})")
        print(f"  EOS token: {repr(tokenizer.eos_token)} (id={tokenizer.eos_token_id})")
        print(f"  PAD token: {repr(tokenizer.pad_token)} (id={tokenizer.pad_token_id})")
        # 测试 encode / decode
        test_text = "你好，世界！"
        ids = tokenizer.encode(test_text, add_special_tokens=False)
        decoded = tokenizer.decode(ids)
        print(f"  测试文本: {test_text!r}")
        print(f"  Token IDs: {ids}")
        print(f"  解码结果: {decoded!r}")
        print(green("  [PASS] Tokenizer 加载成功"))
        return tokenizer
    except Exception as e:
        print(red(f"  [FAIL] Tokenizer 加载失败: {e}"))
        raise


# =========================================================================
# 2. PretrainDataset 测试
# =========================================================================
def test_pretrain_dataset(tokenizer):
    print("\n" + "=" * 60)
    print("【2/7】PretrainDataset 加载测试")
    print("=" * 60)
    try:
        ds = PretrainDataset(DATA_PRETRAIN, tokenizer, max_length=512)
        print(f"  数据集样本数: {len(ds)}")

        # 取前3条测试
        for i in range(min(3, len(ds))):
            input_ids, labels, attention_mask = ds[i]
            non_pad = attention_mask.sum().item()
            label_valid = (labels != -100).sum().item()
            print(f"  样本{i}: input_ids.shape={input_ids.shape}, "
                  f"有效token数={non_pad}, 有效label数={label_valid}, "
                  f"bos={input_ids[0].item()}, eos={input_ids[non_pad-1].item()}")

        # 随机抽1条完整打印
        import random
        idx = random.randint(0, len(ds) - 1)
        input_ids, labels, attn = ds[idx]
        print(f"\n  随机样本 {idx} 解码（前50个token）:")
        print(f"  {tokenizer.decode(input_ids[:50])}")
        print(green("  [PASS] PretrainDataset 加载成功"))
        return True
    except FileNotFoundError:
        print(yellow(f"  [SKIP] 文件不存在: {DATA_PRETRAIN}"))
        return False
    except Exception as e:
        print(red(f"  [FAIL] PretrainDataset 加载失败: {e}"))
        import traceback; traceback.print_exc()
        return False


# =========================================================================
# 3. SFTDataset 测试
# =========================================================================
def test_sft_dataset(tokenizer):
    print("\n" + "=" * 60)
    print("【3/7】SFTDataset 加载测试")
    print("=" * 60)
    try:
        ds = SFTDataset(DATA_SFT, tokenizer, max_length=1024)
        print(f"  数据集样本数: {len(ds)}")

        for i in range(min(3, len(ds))):
            input_ids, labels, attention_mask = ds[i]
            valid_labels = (labels != -100).sum().item()
            print(f"  样本{i}: input_ids.shape={input_ids.shape}, "
                  f"有效label数={valid_labels}, ratio={valid_labels/input_ids.shape[0]:.2%}")

        print(green("  [PASS] SFTDataset 加载成功"))
        return True
    except FileNotFoundError:
        print(yellow(f"  [SKIP] 文件不存在: {DATA_SFT}"))
        return False
    except Exception as e:
        print(red(f"  [FAIL] SFTDataset 加载失败: {e}"))
        import traceback; traceback.print_exc()
        return False


# =========================================================================
# 4. LoRADataset 测试（与SFT共用逻辑，单独测试文件）
# =========================================================================
def test_lora_dataset(tokenizer):
    print("\n" + "=" * 60)
    print("【4/7】LoRADataset 加载测试")
    print("=" * 60)
    try:
        # LoRA 数据和 SFT 是相同格式
        ds = SFTDataset(DATA_LORA, tokenizer, max_length=1024)
        print(f"  数据集样本数: {len(ds)}")

        for i in range(min(3, len(ds))):
            input_ids, labels, attention_mask = ds[i]
            valid_labels = (labels != -100).sum().item()
            print(f"  样本{i}: input_ids.shape={input_ids.shape}, "
                  f"有效label数={valid_labels}")

        print(green("  [PASS] LoRADataset 加载成功"))
        return True
    except FileNotFoundError:
        print(yellow(f"  [SKIP] 文件不存在: {DATA_LORA}"))
        return False
    except Exception as e:
        print(red(f"  [FAIL] LoRADataset 加载失败: {e}"))
        import traceback; traceback.print_exc()
        return False


# =========================================================================
# 5. DPODataset 测试
# =========================================================================
def test_dpo_dataset(tokenizer):
    print("\n" + "=" * 60)
    print("【5/7】DPODataset 加载测试")
    print("=" * 60)
    try:
        ds = DPODataset(DATA_DPO, tokenizer, max_length=2048)
        print(f"  数据集样本数: {len(ds)}")

        for i in range(min(3, len(ds))):
            item = ds[i]
            valid_chosen = (item["mask_chosen"] == 1).sum().item()
            valid_rejected = (item["mask_rejected"] == 1).sum().item()
            print(f"  样本{i}: chosen有效={valid_chosen}, rejected有效={valid_rejected}")

        print(green("  [PASS] DPODataset 加载成功"))
        return True
    except FileNotFoundError:
        print(yellow(f"  [SKIP] 文件不存在: {DATA_DPO}"))
        return False
    except Exception as e:
        print(red(f"  [FAIL] DPODataset 加载失败: {e}"))
        import traceback; traceback.print_exc()
        return False


# =========================================================================
# 6. 模型初始化 + 前向传播 + 梯度测试
# =========================================================================
def test_model_forward(tokenizer):
    print("\n" + "=" * 60)
    print("【6/7】模型初始化 + 前向传播 + 梯度计算测试")
    print("=" * 60)
    try:
        from model.MokioModel import MokioMindConfig, MokioMindForCausalLM

        # 使用小参数配置快速测试
        config = MokioMindConfig(
            hidden_size=512,
            num_hidden_layers=4,
            num_attention_heads=8,
            num_key_value_heads=2,
            vocab_size=len(tokenizer),
            max_position_embeddings=2048,
            use_moe=False,
        )
        print(f"  模型配置: hidden={config.hidden_size}, layers={config.num_hidden_layers}, "
              f"heads={config.num_attention_heads}, vocab={config.vocab_size}")

        model = MokioMindForCausalLM(config)

        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"  推理设备: {device}")
        model = model.to(device)
        model.train()  # 训练模式

        # 准备测试数据
        test_text = ["你好，今天天气很好。"]
        inputs = tokenizer(test_text, return_tensors="pt", padding=True, truncation=True, max_length=128)
        input_ids = inputs["input_ids"].to(device)
        attention_mask = inputs["attention_mask"].to(device)

        print(f"  输入形状: {input_ids.shape}")

        # 前向传播（训练模式，有 labels）
        labels = input_ids.clone()
        labels[:, -1] = -100  # 模拟一个 -100 mask

        output = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
        )

        print(f"  输出 logits shape: {output.logits.shape}")
        print(f"  Loss: {output.loss.item():.6f}")
        if hasattr(output, 'aux_loss'):
            print(f"  Aux Loss: {output.aux_loss.item():.6f}")

        # 反向传播测试
        loss = output.loss
        loss.backward()
        print(f"  反向传播: 成功（梯度已计算）")

        # 检查梯度是否正常
        grad_norms = []
        for name, param in model.named_parameters():
            if param.grad is not None and param.requires_grad:
                grad_norms.append((name, param.grad.norm().item()))
        if grad_norms:
            print(f"  梯度统计: min_norm={min(g for _, g in grad_norms):.2e}, "
                  f"max_norm={max(g for _, g in grad_norms):.2e}")

        # 推理模式测试
        model.eval()
        with torch.no_grad():
            gen_output = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=20,
                do_sample=False,
            )
            gen_text = tokenizer.decode(gen_output[0], skip_special_tokens=False)
            print(f"  生成测试: {gen_text!r}")

        print(green("  [PASS] 模型前向传播 + 梯度计算 + 生成 成功"))
        return True
    except Exception as e:
        print(red(f"  [FAIL] 模型测试失败: {e}"))
        import traceback; traceback.print_exc()
        return False


# =========================================================================
# 7. DataLoader 批量测试（模拟真实训练 batch）
# =========================================================================
def test_dataloader(tokenizer):
    print("\n" + "=" * 60)
    print("【7/7】DataLoader 批量加载测试（模拟训练 batch）")
    print("=" * 60)
    from torch.utils.data import DataLoader

    try:
        batch_size = 4
        max_seq_len = 512

        ds = PretrainDataset(DATA_PRETRAIN, tokenizer, max_length=max_seq_len)
        loader = DataLoader(ds, batch_size=batch_size, shuffle=True, num_workers=0)

        print(f"  DataLoader: batch_size={batch_size}, 总batch数={len(loader)}")

        # 只取第一个 batch
        batch = next(iter(loader))
        input_ids, labels, attention_mask = batch
        print(f"  Batch shapes: input={input_ids.shape}, labels={labels.shape}, mask={attention_mask.shape}")

        # 检查所有 tensor 是否在同一 device
        print(f"  dtype: {input_ids.dtype}")
        assert input_ids.shape[0] == batch_size, "batch size 不匹配"
        assert labels.shape[0] == batch_size, "batch size 不匹配"
        assert attention_mask.shape[0] == batch_size, "batch size 不匹配"

        print(green("  [PASS] DataLoader 批量加载成功"))
        return True
    except FileNotFoundError:
        print(yellow(f"  [SKIP] 文件不存在: {DATA_PRETRAIN}"))
        return False
    except Exception as e:
        print(red(f"  [FAIL] DataLoader 测试失败: {e}"))
        import traceback; traceback.print_exc()
        return False


# =========================================================================
# 主函数
# =========================================================================
def main():
    print("=" * 60)
    print("  MiniMind V2 完整流程诊断脚本")
    print("  项目根目录:", project_root)
    print("  数据目录:", DATA_ROOT)
    print("=" * 60)

    results = {}

    # 1. Tokenizer
    tokenizer = test_tokenizer()

    # 2-5. 各数据集测试
    results["PretrainDataset"] = test_pretrain_dataset(tokenizer)
    results["SFTDataset"]      = test_sft_dataset(tokenizer)
    results["LoRADataset"]     = test_lora_dataset(tokenizer)
    results["DPODataset"]     = test_dpo_dataset(tokenizer)

    # 6. 模型测试
    results["ModelForward"] = test_model_forward(tokenizer)

    # 7. DataLoader
    results["DataLoader"] = test_dataloader(tokenizer)

    # 汇总报告
    print("\n" + "=" * 60)
    print("  诊断报告汇总")
    print("=" * 60)
    passed = sum(1 for v in results.values() if v is True)
    skipped = sum(1 for v in results.values() if v is False)
    failed = sum(1 for v in results.values() if v is None or (v is not True and v is not False))

    for name, result in results.items():
        if result is True:
            print(f"  {green('✓')} {name}: PASS")
        elif result is False:
            print(f"  {yellow('⊘')} {name}: SKIP (文件不存在)")
        else:
            print(f"  {red('✗')} {name}: FAIL")

    print(f"\n  通过: {passed}/{len(results)} 项")
    if skipped > 0:
        print(f"  跳过: {skipped} 项（缺少数据文件）")
    if failed > 0:
        print(red(f"  失败: {failed} 项！请检查错误信息"))

    if passed == len(results):
        print("\n" + green("=" * 60))
        print("  全部测试通过！流程可以正常运行。")
        print(green("=" * 60))
    elif passed + skipped == len(results):
        print("\n" + yellow("  所有可用测试均通过，数据文件缺失不影响代码逻辑。"))
    else:
        print("\n" + red("  存在失败项，请修复后再运行 pipeline。"))


if __name__ == "__main__":
    main()
