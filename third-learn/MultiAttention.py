import torch
import torch.nn as nn
import SelfAttetion


# 多头注意力 即多个自适应因果注意力

# 基于矩阵乘法 进行并行实现

class MultiAttentionWrapper(nn.Module):
    def __init__(self, d_in, d_out, context_len, dropout, num_heads, qkv_bias=False):
        super().__init__()
        self.heads = nn.ModuleList(
            [SelfAttetion.CausalAttention(d_in, d_out, context_len, dropout, qkv_bias) for _ in range(num_heads)])

    def forward(self, x):
        # cat 类比字符串 矩阵拼接
        return torch.cat([head(x) for head in self.heads], dim=-1)


inputs = torch.tensor(
    [[0.43, 0.15, 0.89],
     [0.55, 0.87, 0.66],
     [0.57, 0.85, 0.64],
     [0.22, 0.58, 0.33],
     [0.77, 0.25, 0.10],
     [0.05, 0.80, 0.55]]
)

batch = torch.stack((inputs, inputs), dim=0)
print(batch.shape)

torch.manual_seed(123)
context_len = batch.shape[1]
# 输入维度 和 输出维度
d_in, d_out = 3, 2
mha = MultiAttentionWrapper(d_in, d_out, context_len, 0.0, num_heads=2)
context_vecs = mha(batch)
print("context_vecs: ",context_vecs)
print("context_Vecs ", context_vecs.shape)

#练习 3.2 更改MultiHeadAttentionWrapper(..., num_heads=2)调用的输入参数，使输出上下文向量是二维而不是四维

#4维 是两个 2维度的单头拼接而成 因此 将 dim_out 改为 1即可
