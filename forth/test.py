import tiktoken
import torch
import gpt_learn
import torch.nn as nn

tokenizer = tiktoken.get_encoding("gpt2")
batch = []
txt1 = "Every effort moves you"
txt2 = "Every day holds a"

# 转化为张量
batch.append(torch.tensor(tokenizer.encode(txt1)))
batch.append(torch.tensor(tokenizer.encode(txt2)))

batch = torch.stack(batch, dim=0)

print(batch)

# 初始化实例

torch.manual_seed(123)
model = gpt_learn.DummpyGPTModel(gpt_learn.gpt_config_124M)
logits = model(batch)
print("output shape:", logits.shape)
print(logits)

# 层归一化 示例
torch.manual_seed(123)
batch_example = torch.randn(2, 5)
# 神经网络层layer nn.ReLU()为修正激活函数 将负值设置为0
layer = nn.Sequential(nn.Linear(5, 6), nn.ReLU())
out = layer(batch_example)
print(out)

# 均值 和 方差
mean = out.mean(dim=-1, keepdim=True)
var = out.var(dim=-1, keepdim=True)
print("mean", mean)
print("var", var)

# 进行层归一化操作
out_norm = (out - mean) / torch.sqrt(var)
mean = out_norm.mean(dim=-1, keepdim=True)
var = out_norm.var(dim=-1, keepdim=True)
torch.set_printoptions(sci_mode=False)
print("mean", mean)
print("var", var)

# 测试归一化模块
ln = gpt_learn.LayerNorm(emb_dim=5)
out_ln = ln(batch_example)
mean = out_ln.mean(dim=-1, keepdim=True)
var = out_ln.var(dim=-1, keepdim=True)
print("mean", mean)
print("var", var)
