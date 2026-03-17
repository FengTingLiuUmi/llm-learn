from tkinter.scrolledtext import example

import torch
import torch.nn as nn

# 一个简化的自注意力类
inputs = torch.tensor(
    [[0.43, 0.15, 0.89],
     [0.55, 0.87, 0.66],
     [0.57, 0.85, 0.64],
     [0.22, 0.58, 0.33],
     [0.77, 0.25, 0.10],
     [0.05, 0.80, 0.55]]
)


# 有三个 自适应注意力矩阵 x为输入个数 y为输出个数
class SelfAttentionV1(nn.Module):
    def __init__(self, d_in, d_out):
        super().__init__()
        self.W_query = nn.Parameter(torch.rand(d_in, d_out))
        self.W_key = nn.Parameter(torch.rand(d_in, d_out))
        self.W_value = nn.Parameter(torch.rand(d_in, d_out))

    # 正向传播
    def forward(self, x):
        keys = x @ self.W_key
        queries = x @ self.W_query
        values = x @ self.W_value

        # 得到注意的权重矩阵 未进行归一化 以及缩放
        attn_scores = queries @ keys.T
        # 进行归一化 以及缩放
        attn_weights = torch.softmax(attn_scores / keys.shape[-1] ** 0.5, dim=-1)
        # 最终的注意力
        context_vec = attn_weights @ values
        return context_vec


torch.manual_seed(123)
sa_v1 = SelfAttentionV1(d_in=3, d_out=2)
print(sa_v1.forward(inputs))


# 有三个 自适应注意力矩阵 x为输入个数 y为输出个数
class SelfAttentionV2(nn.Module):
    def __init__(self, d_in, d_out, qkv_bias=False):
        super().__init__()
        # 基于nn.Linear基础化的权重更好
        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)

    # 正向传播
    def forward(self, x):
        keys = self.W_key(x)
        queries = self.W_query(x)
        values = self.W_value(x)

        # 得到注意的权重矩阵 未进行归一化 以及缩放
        attn_scores = queries @ keys.T
        # 进行归一化 以及缩放
        attn_weights = torch.softmax(attn_scores / keys.shape[-1] ** 0.5, dim=-1)
        # 最终的注意力
        context_vec = attn_weights @ values
        return context_vec


torch.manual_seed(789)
v1 = SelfAttentionV1(d_in=3, d_out=2)
v2 = SelfAttentionV2(d_in=3, d_out=2)

print("v2 show:", v2(inputs))

# 练习3.1
v1.W_query = nn.Parameter(v2.W_query.weight.T)
print("v1: q:", v1.W_query)
print("v2: q:", v2.W_query.weight)
v1.W_key = nn.Parameter(v2.W_key.weight.T)
print("v1: k:", v1.W_key)
print("v2: k:", v2.W_key.weight)
v1.W_value = nn.Parameter(v2.W_value.weight.T)
print("v1: v:", v1.W_value)
print("v2: v:", v2.W_value.weight)

# 掩码机制 对于需要预测的部分 屏蔽上下文

queries = v2.W_query(inputs)
keys = v2.W_key(inputs)
attn_scores = queries @ keys.T
# 归一化 和 缩放
attn_weights = torch.softmax(attn_scores / keys.shape[-1] ** 0.5, dim=-1)
print(attn_weights)

# 生成掩码矩阵 即对角线下方为全1 上方为全0
context_length = attn_scores.shape[0]
mask_simple = torch.tril(torch.ones(context_length, context_length))
print(mask_simple)

# 和注意力权重矩阵相乘 对上方进行掩码
masked_result = attn_weights * mask_simple
print("掩码结果 ", masked_result)
# 重新归一化
masked_attn_weights = mask_simple / mask_simple.sum(dim=-1, keepdim=True)
print("掩码后的权重矩阵 ", masked_attn_weights)

# 通过将掩码为设置为 负无穷 从而 在softmax时设置为0
mask = torch.triu(torch.ones(context_length, context_length), diagonal=1)
print("mask :", mask)
# 设置负无穷
masked = attn_scores.masked_fill(mask.bool(), -torch.inf)
print("masked :", masked)
# 进行softmax归一化 即可得到 最终掩码后的 权重矩阵
attn_weights = torch.softmax(masked / keys.shape[-1] ** 0.5, dim=-1)
print(attn_weights)

#dropout  在特定时刻随机忽略某一单元 避免结果过度依赖某一单元 ---->仅在训练时进行
#主要是为了防止过拟合

torch.manual_seed(123)
dropout = torch.nn.Dropout(0.5)
example4Drop = torch.ones(6,6)
print("dropout example:",dropout(example4Drop))
#从结果可以看出 当一半元素被dropout后 为了补充 剩余的元素会被放大 1/0.5 倍
#用概率的角度理解的话 就是概率综合依旧需要为1
print("dropout example:",dropout(attn_weights))
