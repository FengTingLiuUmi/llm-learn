import torch
import torch.nn as nn

#一个简化的自注意力类
inputs = torch.tensor(
    [[0.43,0.15,0.89],
    [0.55,0.87,0.66],
    [0.57,0.85,0.64],
    [0.22,0.58,0.33],
    [0.77,0.25,0.10],
    [0.05,0.80,0.55]]
)

#有三个 自适应注意力矩阵 x为输入个数 y为输出个数
class SelfAttentionV1(nn.Module):
    def __init__(self,d_in,d_out):
        super().__init__()
        self.W_query = nn.Parameter(torch.rand(d_in,d_out))
        self.W_key = nn.Parameter(torch.rand(d_in, d_out))
        self.W_value = nn.Parameter(torch.rand(d_in, d_out))

    #正向传播
    def forward(self,x):
        keys = x @ self.W_key
        queries = x @ self.W_query
        values = x @ self.W_value

        #得到注意的权重矩阵 未进行归一化 以及缩放
        attn_scores = queries @ keys.T
        #进行归一化 以及缩放
        attn_weights = torch.softmax(attn_scores/keys.shape[-1]**0.5,dim=-1)
        #最终的注意力
        context_vec = attn_weights @ values
        return context_vec

torch.manual_seed(123)
sa_v1 = SelfAttentionV1(d_in = 3,d_out=2)
print(sa_v1.forward(inputs))

#有三个 自适应注意力矩阵 x为输入个数 y为输出个数
class SelfAttentionV2(nn.Module):
    def __init__(self,d_in,d_out):
        super().__init__()
        #基于nn.Linear基础化的权重更好
        self.W_query = nn.Linear(d_in,d_out)
        self.W_key = nn.Linear(d_in, d_out)
        self.W_value = nn.Linear(d_in, d_out)

    #正向传播
    def forward(self,x):
        keys = x @ self.W_key
        queries = x @ self.W_query
        values = x @ self.W_value

        #得到注意的权重矩阵 未进行归一化 以及缩放
        attn_scores = queries @ keys.T
        #进行归一化 以及缩放
        attn_weights = torch.softmax(attn_scores/keys.shape[-1]**0.5,dim=-1)
        #最终的注意力
        context_vec = attn_weights @ values
        return context_vec

#练习3.1

v1 = SelfAttentionV1(d_in = 3,d_out=2)
v2 = SelfAttentionV2(d_in = 3,d_out=2)
v1.W_query = nn.Parameter(v2.W_query.weight.T)
print("v1: q:",v1.W_query)
print("v2: q:",v2.W_query.weight)
v1.W_key = nn.Parameter(v2.W_key.weight.T)
print("v1: k:",v1.W_key)
print("v2: k:",v2.W_key.weight)
v1.W_value = nn.Parameter(v2.W_value.weight.T)
print("v1: v:",v1.W_value)
print("v2: v:",v2.W_value.weight)

