import torch
from torch import optim, nn

# 定义Lora网络结构
class LoRA(nn.Module):
    def __init__(self, in_features, out_features, rank):
        super().__init__()
        self.rank = rank
        self.A = nn.Linear(in_features, rank, bias=False)
        self.B = nn.Linear(rank, out_features, bias=False)

        # 矩阵a高斯初始化
        self.A.weight.data.normal_(mean=0.0, std=0.02)
        # 矩阵b全0初始化
        self.B.weight.data.zero_()
    
    def forward(self, x):
        return self.B(self.A(x))

def apply_lora(model, rank=8):
    # 获取模型当前所在的设备（CPU 或 GPU）
    # 确保后续新建的 LoRA 层也在同一个设备上，避免计算时报错。
    device = next(model.parameters()).device

    # 遍历模型中所有的子模块。
    # 筛选条件： 必须是线性层 (nn.Linear)。
    # 注意：移除了方阵限制，使 LoRA 同时作用于 attention 的 q/k/v/o 所有投影层，
    # 这是在 LLM 上应用 LoRA 的标准做法（原代码只作用于 q_proj/o_proj，漏掉了 k_proj/v_proj）。
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            lora = LoRA(module.weight.shape[0], module.weight.shape[1], rank=rank).to(device)
            # setattr(object, name, value) 是 Python 的内置函数，用于给对象设置属性。

            # 参数含义：
            # object: 要设置属性的对象。
            # name: 属性名的字符串。
            # value: 要赋予该属性的值。

            # 在代码中的作用：
            # Python
            # setattr(module, "lora", lora)
            # 这行代码相当于执行了 module.lora = lora。

            # 为什么要用 setattr？
            # 因为 module 是在循环中动态获取的，虽然这里可以直接写点号，但在处理动态变量名或确保兼容性时，setattr 更加显式和灵活。
            # 它将新创建的 LoRA 层挂载到了当前的 nn.Linear 模块上，使其成为该模块的一个子属性。
            setattr(module, "lora", lora)
            original_forward = module.forward

            # 显式绑定
            def forward_with_lora(x, layer1=original_forward, layer2=lora):
                return layer1(x) + layer2(x)

            module.forward = forward_with_lora

def load_lora(model, path):
    device = next(model.parameters()).device
    state_dict = torch.load(path, map_location=device)
    state_dict = {(k[7:] if k.startswith("module.") else k): v for k, v in state_dict.items()}

    for name, module in model.named_modules():
        if hasattr(module, "lora"):
            lora_state = {k.replace(f"{name}.lora.", ""): v for k, v in state_dict.items() if f"{name}.lora." in k}
            module.lora.load_state_dict(lora_state)

def save_lora(model, path):
    raw_model = getattr(model, "_orig_mod", model)
    state_dict = {}
    for name, module in raw_model.named_modules():
        if hasattr(module, "lora"):
            clean_name = name[7:] if name.startswith("module.") else name
            lora_state = {f"{clean_name}.lora.{k}": v for k, v in module.lora.state_dict().items()}
            state_dict.update(lora_state)
    torch.save(state_dict, path)
