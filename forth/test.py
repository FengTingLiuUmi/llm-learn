import tiktoken
import torch
import gpt_learn
import torch.nn as nn
import matplotlib.pyplot as plt

from forth.gpt_learn import ExampleDeepNeuralNetwork

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

# GELU函数和 ReLU函数对比
gelu, relu = gpt_learn.GELU(), nn.ReLU()

# 在 -3 - 3 间 创建 100个数据点
x = torch.linspace(-3, 3, 100)
y_gelu, y_relu = gelu(x), relu(x)
plt.figure(figsize=(8, 3))
for i, (y, label) in enumerate(zip([y_gelu, y_relu], ["GELU", "RELU"]), 1):
    plt.subplot(1, 2, i)
    plt.plot(x, y)
    plt.title(f"{label} activation function")
    plt.xlabel("x")
    plt.ylabel(f"{label} {x}")
    plt.grid(True)
plt.tight_layout()
plt.show()

# FeedForward模块测试

def print_gradintes(model, x):
    output = model(x)  # 向前传播
    target = torch.tensor([[0.]])

    loss = nn.MSELoss()  # 损失函数
    loss = loss(output, target)
    loss.backward()  # 反向传播

    for name, param in model.named_parameters():
        if 'weight' in name:
            print(f"{name} has gradient mean of {param.grad.abs().mean().item()}")


layer_sizes = [3, 3, 3, 3, 3, 1]
sample_input = torch.tensor([1.,0.,-1.])
torch.manual_seed(123)
model_without_shortcut = ExampleDeepNeuralNetwork(layer_sizes,use_shortcut=False)
print_gradintes(model_without_shortcut,sample_input)