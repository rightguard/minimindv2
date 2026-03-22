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

    # 使用显式栈代替递归，避免 named_modules() 的递归深度问题
    # 筛选条件： 必须是线性层 (nn.Linear)。
    # 注意：移除了方阵限制，使 LoRA 同时作用于 attention 的 q/k/v/o 所有投影层，
    # 这是在 LLM 上应用 LoRA 的标准做法（原代码只作用于 q_proj/o_proj，漏掉了 k_proj/v_proj）。

    # 手动使用 _modules 字典遍历，避免 named_modules() 的递归问题
    modules_to_process = [('', model)]
    processed = set()
    max_iterations = 100000  # 安全限制
    iteration_count = 0

    while modules_to_process:
        iteration_count += 1
        if iteration_count > max_iterations:
            raise RuntimeError(f"Module iteration exceeded {max_iterations} iterations. Possible cycle detected.")
        name, module = modules_to_process.pop()
        if id(module) in processed:
            continue
        processed.add(id(module))

        if isinstance(module, nn.Linear):
            lora = LoRA(module.weight.shape[0], module.weight.shape[1], rank=rank).to(device)
            setattr(module, "lora", lora)
            original_forward = module.forward

            def forward_with_lora(x, layer1=original_forward, layer2=lora):
                return layer1(x) + layer2(x)

            module.forward = forward_with_lora

        # 手动遍历 _modules 字典，避免递归
        if not hasattr(module, '_modules'):
            continue
        for child_name, child_module in module._modules.items():
            if child_module is None:
                continue
            if id(child_module) in processed:
                continue
            full_name = f"{name}.{child_name}" if name else child_name
            modules_to_process.append((full_name, child_module))

def _iter_modules(model):
    """迭代遍历模型模块，避免递归深度问题"""
    # 使用显式栈遍历 _modules 字典，避免 named_modules() 的递归问题
    modules_to_process = [('', model)]
    processed = set()
    max_iterations = 100000  # 安全限制，防止意外死循环
    iteration_count = 0
    while modules_to_process:
        iteration_count += 1
        if iteration_count > max_iterations:
            raise RuntimeError(f"Module iteration exceeded {max_iterations} iterations. Possible cycle detected.")
        name, module = modules_to_process.pop()
        if id(module) in processed:
            continue
        processed.add(id(module))
        yield name, module
        # 手动遍历 _modules 字典
        if not hasattr(module, '_modules'):
            continue
        for child_name, child_module in module._modules.items():
            if child_module is None:
                continue
            if id(child_module) in processed:
                continue
            full_name = f"{name}.{child_name}" if name else child_name
            modules_to_process.append((full_name, child_module))

def load_lora(model, path):
    raw_model = getattr(model, "_orig_mod", model)
    raw_model = getattr(raw_model, "module", raw_model)
    device = next(raw_model.parameters()).device
    state_dict = torch.load(path, map_location=device)
    state_dict = {(k[7:] if k.startswith("module.") else k): v for k, v in state_dict.items()}

    for name, module in _iter_modules(raw_model):
        if hasattr(module, "lora"):
            lora_state = {k.replace(f"{name}.lora.", ""): v for k, v in state_dict.items() if f"{name}.lora." in k}
            module.lora.load_state_dict(lora_state)

def save_lora(model, path):
    raw_model = getattr(model, "_orig_mod", model)
    raw_model = getattr(raw_model, "module", raw_model)
    state_dict = {}
    for name, module in _iter_modules(raw_model):
        if hasattr(module, "lora"):
            clean_name = name[7:] if name.startswith("module.") else name
            lora_state = {f"{clean_name}.lora.{k}": v for k, v in module.lora.state_dict().items()}
            state_dict.update(lora_state)
    torch.save(state_dict, path)
