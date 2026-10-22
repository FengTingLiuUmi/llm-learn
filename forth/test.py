import tiktoken
import torch
import gpt_learn
import torch.nn as nn
import matplotlib.pyplot as plt

import Transformer

from forth.Transformer import TransformerBlock, GPTModel
from forth.gpt_learn import LayerNorm


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
model = gpt_learn.DummyGPTModel(gpt_learn.gpt_config_124M)
#logits 模型的输出 对数几率
logits = model(batch)
print("output shape:", logits.shape)
print(logits)
#后处理为文本

# 层归一化 示例 均值为0 方差为1
torch.manual_seed(123)
batch_example = torch.randn(2, 5)
# 神经网络层layer nn.ReLU()为修正激活函数 将负值设置为0
layer = nn.Sequential(nn.Linear(5, 6), nn.ReLU())
out = layer(batch_example)
print('example')
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


#测试激活函数区别

gelu, relu = gpt_learn.GELU(), nn.ReLU()

x = torch.linspace(-3,3,100)
y_gelu, y_relu = gelu(x),relu(x)
plt.figure(figsize = (8,3))
for i , (y, label) in enumerate(zip([y_gelu,y_relu] , ['GELU' , 'RELU']),1) :
    plt.subplot(1,2,i)
    plt.plot(x,y)
    plt.title(f"{label} activation function")
    plt.xlabel("x")
    plt.ylabel(f"{label} (x)")

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


gpt_config_124M = {
    "vocab_size": 50257,  # 词汇表大小
    "context_length": 1024,  # 上下文长度
    "emb_dim": 768,  # 嵌入维度 就是 表示一个词有多少 信息维度
    "n_heads": 12,  # 注意力头的数量
    "n_layers": 12,  # 层数
    "drop_rate": 0.1,  # drop_out率
    "qkv_bias": False  # 偏置
}

torch.manual_seed(123)
x = torch.rand(2, 4 ,768)
block = TransformerBlock(gpt_config_124M)
output = block(x)

#输入维度等于输出 使得可以作为下一个输入
print('input shape:', x.shape)
print('output shape:', output.shape)


torch.manual_seed(123)
model = GPTModel(gpt_config_124M)

out = model(batch)
print('input batch \n' , batch)
print('output shape \n' , out.shape)
print('output  \n' , out)

#GPT2中有权重共享
total_params = sum(p.numel() for p in model.parameters())

#文本生产
# softmax将 logits分数转为概率分布 取最大 即为词元Id 从词汇中选出 作为新的输入，继续生产 。 直到达到上限/生成的是结束符号

start_context = 'Hello , I am'
encoded = tokenizer.encode(start_context)
print('encode: ' , encoded)
encoded_tensor = torch.tensor(encoded).unsqueeze(0)
print('shape:' , encoded_tensor.shape)
model.eval() #禁用dropout 等训练用值
out = Transformer.generate_text_sample(model=model,idx = encoded_tensor , max_new_tokens= 6,context_size= gpt_config_124M['context_length'])
print('output :',out)
print('output len:', len(out[0]))

decoded_text = tokenizer.decode(out.squeeze(0).tolist())

#Hello , I am pedigreeigor66666666 previouslypackPir 没有训练 所以没用
print(decoded_text)




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

