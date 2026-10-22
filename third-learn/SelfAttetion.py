import torch
import torch.nn as nn

# 生成输入

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


class CausalAttention(nn.Module):
    def __init__(self, d_in, d_out, context_len, dropout, qvk_bias=False):
        super().__init__()
        self.d_out = d_out
        self.W_query = nn.Linear(d_in, d_out, bias=qvk_bias)
        self.W_key = nn.Linear(d_in, d_out, bias=qvk_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qvk_bias)
        self.dropout = nn.Dropout(dropout)
        self.register_buffer('mask', torch.triu(torch.ones(context_len, context_len), diagonal=1))

    # 正向传播
    def forward(self, x):
        # 批量大小 token个数 token的词嵌入维度
        batch_size, num_tokens, d_in = x.shape
        keys = self.W_key(x)
        queries = self.W_query(x)
        values = self.W_value(x)

        # 维度 1 和 维度 2转制
        attn_scores = queries @ keys.transpose(1, 2)
        # 填充负无穷 掩码化
        attn_scores.masked_fill(self.mask.bool()[:num_tokens, :num_tokens], -torch.inf)
        attn_weights = torch.softmax(attn_scores / keys.shape[-1] ** 0.5, dim=-1)
        attn_weights = self.dropout(attn_weights)

        context_vec = attn_weights @ values
        return context_vec


torch.manual_seed(123)
context_len = batch.shape[1]
ca = CausalAttention(3, 2, context_len, 0.0)
context_vecs = ca(batch)
print("context_vecs.shape:", context_vecs.shape)
