"""测试 model_lora.py 是否能正常工作"""
import sys
sys.path.insert(0, '.')

import torch
from torch import nn

# 导入 LoRA 模块
from model.model_lora import apply_lora, LoRA

# 创建一个简单的测试模型
class SimpleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = nn.Linear(10, 20)
        self.linear2 = nn.Linear(20, 30)

    def forward(self, x):
        x = self.linear1(x)
        x = self.linear2(x)
        return x

# 测试
model = SimpleModel()

print("Before apply_lora:")
print(f"  linear1.hasattr('lora'): {hasattr(model.linear1, 'lora')}")
print(f"  linear2.hasattr('lora'): {hasattr(model.linear2, 'lora')}")

# 应用 LoRA
apply_lora(model, rank=4)

print("\nAfter apply_lora:")
print(f"  linear1.hasattr('lora'): {hasattr(model.linear1, 'lora')}")
print(f"  linear2.hasattr('lora'): {hasattr(model.linear2, 'lora')}")

# 测试前向传播
x = torch.randn(2, 10)
output = model(x)
print(f"\nForward pass output shape: {output.shape}")

print("\n✅ 测试通过！")
