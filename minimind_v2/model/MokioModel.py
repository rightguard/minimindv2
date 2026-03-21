from turtle import forward
from numpy import repeat
from torch._higher_order_ops.triton_kernel_wrap import Intermediate
from transformers import PretrainedConfig

# 假设一个[2, 10]的token进行tokenize，然后embedding为[2, 10, 512],进行一次dropout
# position_embedding是两个[10, 64]的元组
# 然后进入block层
    # hidden_state[2, 10, 512]，进入rmsnorm和残差不改变形状
    # 进入attention
        # 1.投影,让q投影为attention_head*head_dim
        # 2.k和v投影为kv_head*head_dim
        # 3.然后进行重塑，将最后一维拆开成(nhead,head_dim)
        # 4.之后应用旋转位置编码，逐元素相乘相加，q和k仍为[2,10,8,64]和[2,10,2,64](k和v要重复四份)
        # 5.将q进行transpose，用于计算seq_len*seq_len，即转换为[2,8,10,64]
        # 6.然后k和v repeat k 和v，再transpose复制四次变为[2,8,10,64]
        # 7.进行注意力分数计算，k后两位交换，和q相乘得 (2, 8, 10, 64) @ (2, 8, 64, 10) -> (2, 8, 10, 10)
        # 8.应用因果掩码，把当前token之后的都变为-inf
            # causal_mask : (10, 10)
            # scores = scores + causal_mask.unsqueeze(0).unsqueeze(0): (2, 8, 10, 10) (添加因果掩码)
        # 9.softmax变化，之后再与v相乘 (2, 8, 10, 10) @ (2, 8, 10, 64) -> (2, 8, 10, 64)
        # 10.transpose, reshape, 重新变回[2,10,512]
# 然后进入ffn
    # 输入[2, 10, 512]
    # 按照8/3的比例升维为[2, 10, 1344]
    # 并行，另一部分进入gate和silu，变为门控
    # gate'up，维度不变为[2, 10, 1344]
    # down降维[2,10,512]
# 继续残差处理，进入下一个layer，依旧是[2, 10, 512]
# 整个词表进行映射[2, 10, 6400]，输出logits
    


class MokioMindConfig(PretrainedConfig):
    model_type = "mokiomind"

    def __init__(
        self,
        dropout: float = 0.0,
        bos_token_id: int = 1,
        eos_token_id: int = 2,
        hidden_act: str = "silu",
        hidden_size: int = 512,
        intermediate_size: int = None,
        max_position_embeddings: int = 32768,
        num_attention_heads: int = 8,
        num_hidden_layers: int = 8,
        num_key_value_heads: int = 2,
        vocab_size: int = 6400,
        rms_norm_eps: float = 1e-05,
        rope_theta: int = 1000000,
        inference_rope_scaling: bool = False,
        flash_attention: bool = True,
        ############ MoE ############
        use_moe: bool = False,
        num_experts_per_tok: int = 2,
        n_routed_experts: int = 4,
        n_shared_experts: int = 1,
        scoring_func: str = "softmax",
        aux_loss_alpha: float = 0.01,
        seq_aux: bool = True,
        norm_topk_prob: bool = True,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.dropout = dropout
        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id
        self.hidden_act = hidden_act
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.max_position_embeddings = max_position_embeddings
        self.num_attention_heads = num_attention_heads
        self.num_hidden_layers = num_hidden_layers
        self.num_key_value_heads = num_key_value_heads
        self.vocab_size = vocab_size
        self.rms_norm_eps = rms_norm_eps
        self.rope_theta = rope_theta
        self.inference_rope_scaling = inference_rope_scaling
        self.flash_attention = flash_attention
        self.use_moe = use_moe
        self.num_experts_per_tok = num_experts_per_tok
        self.n_routed_experts = n_routed_experts
        self.n_shared_experts = n_shared_experts
        self.seq_aux = seq_aux
        self.norm_topk_prob = norm_topk_prob
        self.aux_loss_alpha = aux_loss_alpha
        self.scoring_func = scoring_func

        self.rope_scaling = (
            {
                "beta_fast": 32,
                "beta_slow": 1,
                "factor": 16,
                "original_max_position_embeddings": 2048,
                "attention_factor": 1.0,
                "type": "yarn",
            }
            if self.inference_rope_scaling
            else None
        )

import torch
import math
import torch.nn as nn
from torch.nn import init
from typing import Optional, Tuple, List, Union
import torch.nn.functional as F
from transformers.activations import ACT2FN
from transformers import PreTrainedModel, GenerationMixin, PretrainedConfig
from transformers.modeling_outputs import CausalLMOutputWithPast

class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))
    
    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
    
    def forward(self, x):
        return self.weight * self._norm(x.float()).type_as(x)

def precompute_freqs(dim: int, end: int=(32 * 1024), rope_base: float=1e6, rope_scaling:Optional[dict]=None):
    '''
    1. 为什么维度要整除 2 (d/2)？
    在数学上，RoPE 将一个 d 维向量视为 d/2 个复数对。
    假设向量为 x = [x_1, x_2, ..., x_d]
    它被看作：(x_1 + ix_2), (x_3 + ix_4), ..., (x_{d-1} + ix_d) 每个复数对都在一个二维平面上旋转。
    (q和k成一对，因此需要计算d/2个旋转角度)
    因此，我们只需要计算 d/2 个旋转角度（频率）。
    '''
    # 初始化rope频率
    # 1. 初始化标准 RoPE 频率。
    # torch.arange(0, dim, 2) 生成 [0, 2, 4, ... dim-2]
    # 计算出的 freqs 就是标准的 1 / (base ** (2i / d))
    freqs, attn_factor = (1.0 / (rope_base **(torch.arange(0, dim, 2)[:(dim//2)].float()/dim)),1.0)
    
    if rope_scaling is not None:
        orig_max, factor, beta_fast, beta_slow = (
            rope_scaling["original_max_position_embeddings"],
            rope_scaling["factor"],
            rope_scaling["beta_fast"],
            rope_scaling["beta_slow"],
        )

        # 推断的长度大于训练的长度，使用缩放
        if end > orig_max:
            # 波长b到i的映射
            inv_dim = lambda b:(dim * math.log(orig_max/(b*2*math.pi))) / (2*math.log(rope_base))

            # 划分高低维度
            # 索引 0：代表第 1 个频率分量（对应原向量的第 0 和第 1 维）。
            # 这是频率最高、波长最短的维度，负责捕捉“近距离”的细节。
            # 索引 dim // 2 - 1：代表最后 1 个频率分量（对应原向量的最后两个维度）。
            # 这是频率最低、波长最长的维度，负责捕捉“远距离”的关联。
            low, high = (max(math.floor(inv_dim(beta_fast)), 0), min(math.ceil(inv_dim(beta_slow)), dim//2 - 1))

            # 计算缩放因子
            # low之前，ramp为0，在high之后，ramp为1
            # 在low和high之间，ramp为线性插值过渡
            # torch.arange(dim // 2)：生成一个索引序列 [0, 1, 2, ..., d/2-1]，代表频率向量的每一个维度。
            # - low：将起点平移到 low 索引处。索引小于 low 的会变成负数。
            # / max(high - low, 0.001)：归一化处理。将 low 到 high 之间的距离缩放到 [0, 1] 之间。
            # 使用 max(..., 0.001) 是为了防止 high == low 时除以零导致报错。
            # clamp(0, 1)：将结果限制在 [0, 1] 之间。
            ramp = torch.clamp((torch.arange(dim // 2, device=freqs.device).float() - low)/ max(high - low, 0.001), 0, 1,)

            # 6. 频率融合公式：f'(i) = f(i) * ((1-γ) + γ/s)
            # 当 ramp=0 时（高频）：系数为 1，保持原频率不变。
            # 当 ramp=1 时（低频）：系数为 1/factor，即对频率进行线性插值缩放。
            # ramp在0-1之间时：平滑过渡。
            freqs = freqs*(1-ramp+ramp/factor)
    
    # 根据目标长度end，生成位置所有索引
    t = torch.arange(end, device=freqs.device)

    # 计算外积，将t和频率部分相乘，得到每个位置的旋转角度
    freqs = torch.outer(t, freqs).float()
    freqs_cos = (torch.cat([torch.cos(freqs), torch.cos(freqs)], dim=-1) * attn_factor)
    freqs_sin = (torch.cat([torch.sin(freqs), torch.sin(freqs)], dim=-1) * attn_factor)
    return freqs_cos, freqs_sin

# rope旋转位置编码
def apply_rotary_pos_emb(q, k, cos, sin, position_ids=None, unsqueeze_dim=1):
    '''
    数学原理：欧拉公式与旋转矩阵

    在复数平面上，将一个向量 x 旋转 theta 角度，等同于乘以 e^(i*theta)：
    (a + bi) * (cosθ + i*sinθ) = (a*cosθ - b*sinθ) + i*(a*sinθ + b*cosθ)

    对应到实数空间，如果一个二维向量是 [a, b]，旋转后的向量就是：
    [a', b'] = [cosθ, -sinθ; sinθ, cosθ] * [a, b]
             = [a*cosθ - b*sinθ, b*cosθ + a*sinθ]

    公式解读：
    - 第一项：a 和 b 分别乘以 cosθ（保持原位）
    - 第二项：-b 和 a 分别乘以 sinθ（位置互换且变号）
    '''
    # [a, b] -> [-b, a]
    def rotate_half(x):
        return torch.cat((-x[..., x.shape[-1]//2:], x[..., :x.shape[-1]//2]), dim=-1)
    
    q_embed = (q * cos.unsqueeze(unsqueeze_dim)) + (rotate_half(q) * sin.unsqueeze(unsqueeze_dim))
    k_embed = (k * cos.unsqueeze(unsqueeze_dim)) + (rotate_half(k) * sin.unsqueeze(unsqueeze_dim))
    return q_embed, k_embed

def repeat_kv(x:torch.Tensor, n_rep: int) -> torch.Tensor:
    '''
    这段代码就是 GQA（分组查询注意力） 实现中的关键：KV 广播（Broadcasting）。
    它的作用是：当你只有少量 KV 头（比如 8 个）但有很多 Q 头（比如 32 个）时
    通过“分身术”让每个 KV 头重复出现，从而能与 Q 头进行一对一的点积运算。


    '''
    # n_rep：重复次数
    # slen 是 Sequence Length 的缩写，意为 “序列长度”。
    # 它表示当前输入模型的一批数据中，包含多少个 Token（词元）。
    bs, slen, num_key_value, head_dim = x.shape
    if n_rep == 1:
        return x
    
    return x[:, :, :, None, :].expand(bs, slen, num_key_value, n_rep, head_dim).reshape(bs, slen, num_key_value * n_rep, head_dim)

class Attention(nn.Module):
    def __init__(self, args:MokioMindConfig):
        super().__init__()
        self.args = args
        
        self.num_key_value_heads = (
            # 简单来说，这行代码的作用是：如果用户没指定使用 GQA（分组注意力）
            # 代码就自动退回到传统的 MHA（多头注意力）。
            args.num_attention_heads
            if args.num_key_value_heads is None 
            else args.num_key_value_heads
        )

        assert args.num_attention_heads % self.num_key_value_heads == 0

        self.n_local_heads = args.num_attention_heads
        self.n_local_kv_heads = self.num_key_value_heads
        self.n_rep = args.num_attention_heads // self.num_key_value_heads

        # head_dim = hidden_size / num_attention_heads
        # hidden_size：整个句子的特征总量（比如 4096）。
        # num_attention_heads：并行处理的“头”数（比如 32 个头）。
        # head_dim：每个头分配到的维度（4096 / 32 = 128）。
        # 物理意义：我们将一个宏大的特征空间（hidden_size）切分成许多个小的空间（head_dim），让不同的“头”在这些小空间里并行工作，有的看语法，有的看逻辑。
        self.head_dim = args.hidden_size // args.num_attention_heads

        # 虽然在大多数标准模型（如 Llama 8B）中，num_heads * head_dim 确实等于 hidden_size，但从代码逻辑上讲，它们代表的物理意义不同：
        # 第一个参数 (hidden_size)：输入向量的维度（输入的特征宽度）。
        # 第二个参数 (num_heads * head_dim)：输出向量的维度（投影后的总特征宽度）。
        # 通过这样写，代码明确表达了：“我要把输入特征投影到由 N 个头组成的空间里”。

        # 2. 核心原因：支持“非对称”投影
        # 这是最关键的一点。在某些变体架构中，投影后的维度并不一定等于输入的维度。
        # 标准 Transformer：投影后的维度 = hidden_size。
        # 某些优化架构：为了减少计算量或增加特定头的表达力，开发者可能会让投影后的总维度大于或小于 hidden_size。
        # 如果代码直接写成：
        # nn.Linear(args.hidden_size, args.hidden_size)
        # 那么这个模型就被“焊死”了，无法通过修改头数或每个头的维度来改变输出形状。而写成 args.num_attention_heads * self.head_dim，则赋予了模型配置上的灵活性。
        self.q_proj = nn.Linear(args.hidden_size, args.num_attention_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(args.hidden_size, args.num_key_value_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(args.hidden_size, args.num_key_value_heads * self.head_dim, bias=False)

        self.o_proj = nn.Linear(args.num_attention_heads * self.head_dim, args.hidden_size, bias=False)

        self.attn_dropout = nn.Dropout(args.dropout)
        self.resid_dropout = nn.Dropout(args.dropout)
        self.dropout = args.dropout
        self.flash = (hasattr(nn.functional, "scaled_dot_product_attention") and self.args.flash_attention)

    
    def forward(self, x, position_embeddings: Tuple[torch.Tensor, torch.Tensor],
                past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
                use_cache=False,
                attention_mask: Optional[torch.Tensor] = None,):

        bsz, seq_len, _ = x.shape

        # 投影计算qkv
        xq, xk, xv = self.q_proj(x), self.k_proj(x), self.v_proj(x)

        # 把输入拆分为多个头，用view
        xq = xq.view(bsz, seq_len, self.n_local_heads, self.head_dim)
        xk = xk.view(bsz, seq_len, self.n_local_kv_heads, self.head_dim)
        xv = xv.view(bsz, seq_len, self.n_local_kv_heads, self.head_dim)

        # q和k使用rope
        cos, sin = position_embeddings
        xq, xk = apply_rotary_pos_emb(xq, xk, cos, sin)

        # 对于kv，使用repeat_kv，（注意kv cache）
        if past_key_value is not None:
            xk = torch.cat([past_key_value[0], xk], dim=1)
            xv = torch.cat([past_key_value[1], xv], dim=1)
        past_kv = (xk, xv) if use_cache else None

        xq, xk, xv = (xq.transpose(1, 2), repeat_kv(xk, self.n_rep).transpose(1, 2), repeat_kv(xv, self.n_rep).transpose(1, 2))

        if (
            self.flash 
            and (seq_len > 1) 
            and (past_key_value is None)
            and (attention_mask is None or torch.all(attention_mask == 1))
        ):
            output = F.scaled_dot_product_attention(xq,
                                                    xk,
                                                    xv,
                                                    dropout_p=self.dropout if self.training else 0.0,
                                                    is_causal=True)
        else:
            scores = (xq @ xk.transpose(-2, -1)  / math.sqrt(self.head_dim))
            scores[:, :, :, -seq_len:] += torch.triu(
                torch.full((seq_len, seq_len), float("-inf"), device=scores.device),
                diagonal=1,
            )

            if attention_mask is not None:
                extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
                extended_attention_mask = (1.0 - extended_attention_mask) * -1e9
                scores = scores + extended_attention_mask
            
            scores = F.softmax(scores.float(), dim=-1).type_as(xq)
            scores = self.attn_dropout(scores)
            output = scores @ xv
        
        output = output.transpose(1, 2).reshape(bsz, seq_len, -1)
        output = self.resid_dropout(self.o_proj(output))
        return output, past_kv

class FeedForward(nn.Module):
    def __init__(self, config: MokioMindConfig):
        super().__init__()

        # 动态计算中间层维度
        # 8 / 3：这是一个“黄金比例”。
        # 在 SwiGLU 结构中，为了保持总参数量与传统 MLP（4倍 hidden_size）一致，中间层维度通常设为 2.66 倍。
        # 64 * ((... + 63) // 64)：这是一个向上取整到 64 倍数的操作。
        # 目的是为了显存对齐（Memory Alignment），让 GPU 的 Tensor Core 计算时效率更高。
        if config.intermediate_size is None:
            intermediate_size = int(config.hidden_size * 8 / 3)
            config.intermediate_size = 64 * ((intermediate_size + 64 - 1) // 64)

        self.gate_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.down_proj = nn.Linear(config.intermediate_size, config.hidden_size, bias=False)
        self.up_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.dropout = nn.Dropout(config.dropout)
        self.act_fn = ACT2FN[config.hidden_act]

    def forward(self, x):
        # 这是全篇最关键的一行，它实现了逐元素相乘（Element-wise product）：
        # 左半部分 self.act_fn(self.gate_proj(x))：就像一个过滤器，输出 0 到 1 之间的值（SiLU 会略微超出这个范围）。
        # 右半部分 self.up_proj(x)：是原始提取出的特征。
        # 相乘：只有“门控”打开的位置，其对应的特征才能流向下一层。
        gated = self.act_fn(self.gate_proj(x)) * self.up_proj(x)
        return self.dropout(self.down_proj(gated))


class MoEGate(nn.Module):
    def __init__(self, config: MokioMindConfig):
        super().__init__()
        self.config = config

        '''
        1. 专家总数 (Total Experts, 记作 N)这是指你在模型层中一共部署了多少个独立的专家网络（通常是 FFN 层）。
        性质： 它决定了模型的总参数量。作用： 专家总数越多，模型理论上能存储的知识就越广、越深。
        比如 DeepSeek-V3 有 256 个专家，Mixtral 8x7B 有 8 个专家。
        存储代价： 专家总数越多，占用的显存就越大（因为所有专家都必须加载进显存）。
        
        2. 路由专家数量 (Active Experts / Top-K, 记作 K)
        这是指对于每一个输入的词（Token），门控网络（Router）最终决定激活并参与计算的专家数量。
        性质： 它决定了模型的计算成本 (FLOPs)。
        作用： 无论总共有多少专家，每次推理只选 K 个。
        最常见的设置是 Top-1 或 Top-2。计算代价： 推理速度只取决于 K的大小。
        即使你总共有 1000 个专家，如果 K=2，计算延迟也只相当于一个比 2 * FFN 稍大一点的稠密模型。
        '''
        # 每个token路由的专家数量
        self.top_k = config.num_experts_per_tok

        # 路由专家数量
        self.n_routed_experts = config.n_routed_experts

        # 打分函数选择
        self.scoring_func = config.scoring_func

        # α是控制辅助损失的权重
        self.alpha = config.aux_loss_alpha

        # 决定是否对序列进行辅助损失计算
        self.seq_aux = config.seq_aux

        self.norm_topk_prob = config.norm_topk_prob
        self.gating_dim = config.hidden_size

        # 创建权重
        self.weight = nn.Parameter(
            torch.empty((self.n_routed_experts, self.gating_dim))
        )
        self.reset_parameters()

    def reset_parameters(self) -> None:
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))

    def forward(self, hidden_states):
        # moe时只看token值不关心其他位置，所以可以合并bsz和seqlen
        bsz, seq_len, h = hidden_states.shape
        hidden_states = hidden_states.view(-1, h)

        # 使用linear映射计算出每个token对于各个expert的logits
        logits = F.linear(hidden_states, self.weight, None)

        # 合并之后，对权重进行线性映射，得到每个token对于各个expert的logits
        # 使用softmax得到每个token对于各个expert的分数
        if self.scoring_func == "softmax":
            scores = logits.softmax(dim=-1)
        else:
            raise NotImplementedError(
                f"insupportable scoring function for MoE gating: {self.scoring_func}"
            )
        
        # 第一种方法，序列级别的aux——loss
        # 第二种方法，批次级别的aux——loss
        topk_weight, topk_idx = torch.topk(scores, k=self.top_k, dim=-1, sorted=False)

        # 归一化
        # 当选中多个专家时，需要对选中专家的权重归一化
        # 目的：确保每个token对多个专家的权重和为1，避免权重累积过大或过小
        if self.top_k > 1 and self.norm_topk_prob:
            denominator = topk_weight.sum(dim=-1, keepdim=True) + 1e-20
            topk_weight = topk_weight / denominator

        # 计算辅助损失（仅训练模式）
        # 首先判断是在训练模式下且alpha大于0
        # 辅助损失的作用：确保附在均衡，防止大多的token仅流向少量的专家

        if self.training and self.alpha > 0.0:
            scores_for_aux = scores    # 保留原始分数用于计算辅助损失
            aux_topk = self.top_k    # 辅助损失的专家数量

            # 将topk_idx从[bsz*seq_len, top_k]变为[bsz, seq_len, top_k]
            topk_idx_for_aux_loss = topk_idx.view(bsz, -1)    # 辅助损失的专家索引

            # 序列级别的辅助损失
            if self.seq_aux:
                # 将scores_for_aux从[bsz*seq_len, n_routed_experts]变为[bsz, seq_len, n_routed_experts]
                scores_for_seq_aux = scores_for_aux.view(bsz, seq_len, -1)    # 辅助损失的专家分数

                # 统计每个batch每个专家被选中的次数
                # ce[bsz, n_routed_experts]，表示每个专家被选中的计数
                ce = torch.zeros(bsz, self.n_routed_experts, device=hidden_states.device)    # 辅助损失的专家计数

                # scatter_add:根据topk_idx_for_aux_loss的值，将1累加到ce中对应的位置
                # 例如topk_idx_for_aux_loss为[0] = [1, 3]，则ce[0, 1]和ce[0, 3]都加1
                ce.scatter_add_(1, topk_idx_for_aux_loss, torch.ones(bsz, seq_len * aux_topk, device=hidden_states.device),).div_(seq_len * aux_topk / self.n_routed_experts)    # 辅助损失的专家计数归一化
                aux_loss = (ce * scores_for_seq_aux.mean(dim=1)).sum(dim=1).mean() * self.alpha



            else:
                # 将topk_idx展平：[bsz, seq_len, top_k] -> [bsz*seq_len*top_k]
                # 转换为one_hot编码，得到 [bsz*seq_len*top_k, n_routed_experts]
                # 举个例子：如果n_routed_experts=5，topk_idx_for_aux_loss包含值0-4
                # 那么one_hot编码会将这些值转换为对应位置为1，其余位置为0的向量
                # 批次级别的辅助损失
                # 将 topk 的索引转为 one-hot 编码
                mask_ce = F.one_hot(topk_idx_for_aux_loss.view(-1), num_classes=self.n_routed_experts)

                # # 计算所有 Token 中，每个专家被选中的比例
                ce = mask_ce.float().mean(0)
                Pi = scores_for_aux.mean(0)    # 计算每个专家被选中的分数的平均值
                fi = ce * self.n_routed_experts    # 计算每个专家被选中的次数的平均值乘以专家总数
                aux_loss = (Pi * fi).sum() * self.alpha    # 计算辅助损失

                # # 归一化，fi 代表每个专家被选中的“强度”
                fi = ce * self.n_routed_experts
                aux_loss = (Pi * fi).sum() * self.alpha
        else:
            aux_loss = scores.new_zeros(1).squeeze()
        return topk_idx, topk_weight, aux_loss



class MoEFeedForward(nn.Module):
    def __init__(self, config: MokioMindConfig):
        super().__init__()
        self.config = config
        # 路由专家层
        self.experts = nn.ModuleList(
            [FeedForward(config) for _ in range(config.n_routed_experts)]
        )

        # 门控层
        self.gate = MoEGate(config)
        if config.n_shared_experts > 0:
            # 共享专家层
            self.shared_experts = nn.ModuleList(
                [FeedForward(config) for _ in range(config.n_shared_experts)]
            )

    def forward(self, x):
        # 保留原有的x作为残差
        identity = x
        orig_shape = x.shape
        bsz, seq_len, h = orig_shape

        # 使用门控机制选择专家
        topk_weight, topk_idx, aux_loss = self.gate(x)
        # 展开x以便处理
        x = x.view(-1, x.shape[-1])

        flat_topk_idx = topk_idx.view(-1)

        if self.training:
            # 按照定义的num_experts_per_tok重复输入token
            # 每个token安排num_experts_per_tok个专家处理
            x = x.repeat_interleave(self.config.num_experts_per_tok, dim=0)

            # y是空张量，与x形状相同
            y = torch.empty_like(x, dtype=x.dtype)

            # 遍历每个专家
            for i, expert in enumerate(self.experts):
                # 找到所有指向专家i的token
                # 然后将这些token输入专家i进行处理
                # 最后将结果放回y对应位置
                expert_out = expert(x[flat_topk_idx == i])
                if expert_out.shape[0] > 0:
                    y[flat_topk_idx == i] = expert_out.to(y.dtype)
                else:
                    y[flat_topk_idx == i] = expert_out.to(y.dtype) + 0*sum(p.sum() for p in expert.parameters())
            
            # 加权求和
            # 最后的y意义是每个token经过专家处理后的加权结果
            y = (y.view(*topk_weight.shape, -1) * topk_weight.unsqueeze(-1)).sum(dim=-1)
            y = y.view(orig_shape)
        # 如果是推理阶段
        else:
            y = self.moe_infer(x, flat_topk_idx, topk_weight.view(-1, 1)).view(*orig_shape)
        if self.config.n_shared_experts > 0:# 处理共享专家
            for expert in self.shared_experts:
                y = y + expert(identity)
        self.aux_loss = aux_loss
        return y
    
    @torch.no_grad()
    # MoE推理方法
    def moe_infer(self, x, flat_expert_indices, flat_expert_weights):
        # 使用cache，创建一个和x形状相同的零张量
        expert_cache = torch.zeros_like(x)
        # 对专家索引进行排序，最后是[0,0,0,1,1,2,2,2,...]这样的顺序
        # 分拣
        idxs = flat_expert_indices.argsort()
        # 统计每个专家被分配到的token数量
        # 打包
        tokens_per_expert = flat_expert_indices.bincount().cpu().numpy().cumsum(0)
        # 计算每个token对应的专家索引
        token_idxs = idxs // self.config.num_experts_per_tok
        # 对每个打包好的包进行处理
        for i, end_idx in enumerate(tokens_per_expert):
            # 计算当前包的起始位置
            start_idx = 0 if i == 0 else tokens_per_expert[i - 1]
            if start_idx == end_idx:
                continue
            # 取出当前包对应的专家
            expert = self.experts[i]
            # 取出token对应的原始id
            exp_token_idx = token_idxs[start_idx:end_idx]
            # 取出token对应的数据
            expert_tokens = x[exp_token_idx]
            # 计算专家输出，一次性处理当前包的所有token
            expert_out = expert(expert_tokens).to(expert_cache.dtype)
            # 加权
            expert_out.mul_(flat_expert_weights[idxs[start_idx:end_idx]])
            # 将结果散点加到缓存中对应位置
            expert_cache.scatter_add_(
                0, exp_token_idx.view(-1, 1).repeat(1, x.shape[-1]), expert_out
            )

        return expert_cache



class MokioMindBlock(nn.Module):
    def __init__(self, layer_id: int, config: MokioMindConfig):
        super().__init__()
        self.num_attention_heads = config.num_attention_heads
        self.hidden_size = config.hidden_size
        self.head_dim = config.hidden_size // config.num_attention_heads
        self.self_attention = Attention(config)

        self.layer_id = layer_id
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        self.mlp = (FeedForward(config) if not config.use_moe else MoEFeedForward(config))
    
    def forward(self, hidden_states, position_embeddings: Tuple[torch.Tensor, torch.Tensor],
                past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
                use_cache=False,
                attention_mask: Optional[torch.Tensor] = None,):
                res = hidden_states

                hidden_states, present_key_value = self.self_attention(
                    self.input_layernorm(hidden_states),  # pre-norm
                    position_embeddings,
                    past_key_value,
                    use_cache,
                    attention_mask,
                )

                hidden_states = res + hidden_states
                hidden_states = hidden_states + self.mlp(self.post_attention_layernorm(hidden_states))

                return hidden_states, present_key_value
    
class MokioMindModel(nn.Module):
    def __init__(self, config: MokioMindConfig):
        super().__init__()
        self.config = config
        self.vocab_size, self.num_hidden_layers = (config.vocab_size, config.num_hidden_layers)

        # 把token转为向量
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.dropout = nn.Dropout(config.dropout)

        self.layers = nn.ModuleList(
            [MokioMindBlock(l, config) for l in range(self.num_hidden_layers)]
        )
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        # rope预计算
        freqs_cos, freqs_sin = precompute_freqs(
            dim=config.hidden_size // config.num_attention_heads,
            end=config.max_position_embeddings,
            rope_base=config.rope_theta,
            rope_scaling=config.rope_scaling,
        )

        self.register_buffer("freqs_cos", freqs_cos, persistent=False)
        self.register_buffer("freqs_sin", freqs_sin, persistent=False)

    
    def forward(self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None,
        use_cache: bool = False,
        **kwargs,):
        # input_ids: [bsz, seq_len]
        batch_size, seq_length = input_ids.shape

        if hasattr(past_key_values, "layers"):
            past_key_values = None
        

        # past_key_values：是一个列表，存放了每一层（Layer）之前的 K 和 V。
        # 逻辑：
        # 如果是第一步（Prefill）：你刚输入 Prompt，此时没有旧缓存，传入的是 None。
        # 代码就会生成一个全为 None 的列表（长度等于层数），代表“历史记录为空”。
        # 如果是后续步（Decode）：你已经生成了一些词，此时会传入上一轮保存的缓存。
        # 代码就会直接使用它，不再初始化。
        past_key_values = past_key_values or [None] * len(self.layers)

        # 计算start_pos：如果存在past，则start_pos为已有past序列长度
        # 拆解：past_key_values[0][0].shape[1]
        # past_key_values[0]：取出第一层的缓存。
        # past_key_values[0][0]：取出第一层缓存里的 Key 矩阵。
        # .shape[1]：在你的代码维度 [bsz, slen, num_heads, head_dim] 中，维度 1 就是序列长度（slen）。
        # 所以这行代码是在问：“缓存里已经存了多少个词了？”
        start_pos = (
            past_key_values[0][0].shape[1] if past_key_values[0] is not None else 0
        )

        # embedding + dropout
        hidden_states = self.dropout(self.embed_tokens(input_ids)) # [bsz, seq_len, hidden_size]

        position_embeddings = (
            self.freqs_cos[start_pos:start_pos + seq_length],
            self.freqs_sin[start_pos:start_pos + seq_length],
        )

        # 列表实现kv缓存
        presents = []
        for layer_idx, (layer, past_key_value) in enumerate(zip(self.layers, past_key_values)):
            hidden_states, present = layer(
                hidden_states,
                position_embeddings,
                past_key_value,
                use_cache,
                attention_mask,
            )
            presents.append(present)
        
        hidden_states = self.norm(hidden_states)

        aux_loss = sum(
            [
                layer.mlp.aux_loss for layer in self.layers if isinstance(layer.mlp, MoEFeedForward)
            ],
            hidden_states.new_zeros(1).squeeze(),
        )

        return hidden_states, presents, aux_loss

class MokioMindForCausalLM(PreTrainedModel, GenerationMixin):
    config_class = MokioMindConfig
    def __init__(self, config: MokioMindConfig):
        super().__init__(config)
        self.model = MokioMindModel(config)
        
        # 语言头
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # 权重共享
        # 输出层的权重和嵌入层的权重共享  
        '''
        Embedding 层（入口）：任务是将 Token ID 映射到一个 hidden_size 维度的向量。
        它在学习“这个词是什么意思”。LM Head 层（出口）：任务是将隐藏状态向量映射回 vocab_size 维度的概率分布。
        它在学习“下一个该说哪个词”。
        为什么可以共享？如果在语义空间里，“苹果”和“梨”这两个词的向量距离很近（因为它们都是水果）
        那么在预测下一个词时，如果上下文是“我吃了一个...”，模型预测出“苹果”和“梨”的概率也理应都很高。
        通过权重共享，我们强制让输入表示和输出预测使用同一套逻辑。
        
        2. 权重共享的核心优势
        A. 减少参数量，节省显存在你的配置中，如果 vocab_size = 6400，hidden_size = 512
        单独的 Embedding 层参数：6400 * 512 = 3.27M
        单独的 LM Head 层参数：512 * 6400 = 3.27M
        共享后：直接省掉了 327 万个参数（约 13MB 显存）。
        对于参数量较小的 MiniMind 或 MokioMind 来说，这个比例非常可观。
        
        B. 加速收敛与增强泛化如果不共享，Embedding 只有在作为输入时才被更新，LM Head 只有在作为输出预测时才被更新。
        共享后，每一个 Token 的权重既作为输入接受训练，又作为输出预测接受反馈。训练效率翻倍，模型能更快地学习到稳定的词向量分布。
        '''
        self.model.embed_tokens.weight = self.lm_head.weight

    def forward(self, input_ids: Optional[torch.Tensor] = None,
                      attention_mask: Optional[torch.Tensor] = None,
                      labels: Optional[torch.Tensor] = None,
                      past_key_values: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None,
                      use_cache: bool = False,
                      logits_to_keep: Union[int, torch.Tensor] = 0,
                      **args,):
        
        hidden_states, past_key_values, aux_loss = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            use_cache=use_cache,
            **args,
        )

        # 如果logits to keep是整数，那就保留最后n个位置
        # 作用：生成的时候只需要最后的logits来预测下一个token

        # 1. 场景分析：为什么要“切片”？
        # A. 训练场景 (logits_to_keep=0 或 False)
        # 在训练时，我们需要对整个序列进行 Loss 计算。比如输入 [我, 是, 谁]，我们要预测 [是, 谁, <eos>]。
        # 此时我们需要全量的 Logits。
        # hidden_states 形状：[bsz, seq_len, hidden_size]
        # logits 形状：[bsz, seq_len, vocab_size]

        # B. 生成场景 (logits_to_keep=1)
        # 在推理（生成）时，你已经有了之前的 KV Cache，本次 input_ids 通常只有一个新词。
        # 你只想知道下一个词是什么。
        # 此时，计算序列中间位置的 Logits 纯属浪费 GPU 算力。
        # 我们只需要 hidden_states 的最后一个时间步。
        # logits 形状：[bsz, 1, vocab_size]

        # 1. slice 的基本语法
        # slice(start, stop, step) 接收三个参数，逻辑和列表索引 [start:stop:step] 完全一致。
        # start: 起始位置（包含）。
        # stop: 结束位置（不包含）。如果是 None，表示一直截取到最后。
        # step: 步长。

        # 4. 形状（Shape）的变化直观图解
        # 让我们看看在三维张量（Tensor）中发生了什么：
        # 假设 hidden_states 形状是 [1, 5, 512] (Batch=1, 长度=5, 维度=512)
        # 执行 slice_indices = slice(-1, None)
        # 执行 hidden_states[:, slice_indices, :]：

        # 第 0 维（Batch）：全部保留。
        # 第 1 维（Sequence）：只切下最后一个方块。
        # 第 2 维（Hidden）：全部保留。

        # 结果形状：变成 [1, 1, 512]。

        # logits_to_keep 是关键变量：
        # 训练模式下：通常设置为 0。slice(0, None) 相当于 [:]，意思是从第 0 位到最后，全都要。
        # 生成模式下：通常设置为 1。slice(-1, None) 相当于 [-1:]，意思是只要最后一个词。
        slice_indices = (
            slice(-logits_to_keep, None)
            if isinstance(logits_to_keep, int)
            else logits_to_keep   # 如果传入的已经是切片对象，直接用
        )

        # 这是在三维张量 [Batch, Seq_len, Hidden_size] 上动刀：: 
        # 第 0 维（Batch）全要，不切。
        # slice_indices：第 1 维（序列长度）执行截取。
        # 如果是生成模式，这里就把 N个词的特征变成了 1 个词 的特征。
        # 第 2 维（隐藏维度）全要，特征信息必须完整保留。

        # 3. 第三步：形状流转（最终成像）假设我们的 vocab_size 是 6400，hidden_size 是 512。
        # 我们来看两种状态下的形状变化：
        # 状态 A：训练（Training）我们需要预测整句话，计算所有位置的 Loss。
        # 输入 hidden_states: [1, 5, 512] (1个句子，5个词，每个词512维)
        # 切片操作: slice(0, None) -> 形状依然是 [1, 5, 512]
        # 经过 lm_head: 每一个词都投影到 6400 维的词表空间。
        # 最终 logits 形状: [1, 5, 6400] (5个词，每个词都有对应的预测概率)

        # 状态 B：推理/生成（Inference）我们只需要预测下一个词。
        # 输入 hidden_states: [1, 5, 512]切片操作: 
        # slice(-1, None) -> 形状变为 [1, 1, 512] (只剩下最后一个词的特征)
        # 经过 lm_head: 只有这一个词投影到词表空间。
        # 最终 logits 形状: [1, 1, 6400] (只有最后一个位置的预测概率)
        logits = self.lm_head(hidden_states[:, slice_indices, :])

        loss = None
        if labels is not None:
            # 第一步：Logits 砍掉最后一位
            # 操作：去掉序列中最后一个时间步产生的预测结果。
            # 原因：最后一个词产生的预测没有对应的“下一个词”作为标签，所以它是没用的。
            # 形状变化：[Batch, Seq_Len, Vocab] -> [Batch, Seq_Len - 1, Vocab]。
            shift_logits = logits[..., :-1, :].contiguous()


            # 操作：去掉序列中的第一个词。
            # 原因：第一个词 A 是作为输入喂给模型的，没有谁去预测 A，我们是从 A 开始预测 B 的。
            # 形状变化：[Batch, Seq_Len] -> [Batch, Seq_Len - 1]。
            # 对齐效果：现在 shift_logits 的第 0 位（对 A 的预测）和 shift_labels 的第 0 位（真实词 B）完美对齐了。
            shift_labels = labels[..., 1:].contiguous()
            loss = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),  # 展平为 [N, Vocab]
                shift_labels.view(-1),  # 展平为 [N]

                # ignore_index=-100：这是一个非常关键的参数。在数据处理时，为了补齐长度（Padding），我们会填充一些无意义的 Token。
                # 我们将这些无效位置的 Label 设为 -100，计算 Loss 时模型就会自动跳过它们，不让填充位干扰梯度更新。
                ignore_index=-100,
            )
        
        output = CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=past_key_values,
            hidden_states=hidden_states,
        )
        output.aux_loss = aux_loss
        return output
        