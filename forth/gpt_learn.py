import torch
import torch.nn as nn

gpt_config_124M = {
    "vocab_size": 50257,  # 词汇表大小
    "context_length": 1024,  # 上下文长度
    "emb_dim": 768,  # 嵌入维度 就是 表示一个词有多少 信息维度
    "n_heads": 12,  # 注意力头的数量
    "n_layers": 12,  # 层数
    "drop_rate": 0.1,  # drop_out率
    "qkv_bias": False  # 偏置
}


class DummyGPTModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        # token层 每个词对应的
        self.tok_emb = nn.Embedding(cfg["vocab_size"], cfg["emb_dim"])
        # token嵌入层 上下文对应的信息
        self.pos_emb = nn.Embedding(cfg["context_length"], cfg["emb_dim"])
        # dropout
        self.drop_emb = nn.Dropout(cfg["drop_rate"])
        # transformer块
        self.trf_blocks = nn.Sequential(
            *[DummyTransformerBlock(cfg)
              for _ in range(cfg["n_layers"])]
        )
        # 归一化层
        self.final_norm = DummyLayerNorm(cfg["emb_dim"])
        # 线性输出层
        self.out_head = nn.Linear(
            cfg["emb_dim"], cfg["vocab_size"], bias=False
        )

    def forward(self, in_idx):
        batch_size, seq_len = in_idx.shape
        tok_embeds = self.tok_emb(in_idx)
        pos_embeds = self.pos_emb(
            torch.arange(seq_len, device=in_idx.device)
        )
        x = tok_embeds + pos_embeds
        x = self.drop_emb(x)
        x = self.trf_blocks(x)
        x = self.final_norm(x)
        logits = self.out_head(x)
        return logits


class DummyTransformerBlock(nn.Module):
    def __init__(self, cfg):
        super().__init__()

    def forward(self, x):
        return x


class DummyLayerNorm(nn.Module):
    def __init__(self, normalized_shape, eps=1e-5):
        super().__init__()

    def forward(self, x):
        return x


class LayerNorm(nn.Module): #层归一化 模块
    def __init__(self, emb_dim):
        super().__init__()
        # 小常量 防止除零错误
        self.eps = 1e-5
        self.scale = nn.Parameter(torch.ones(emb_dim))
        self.shift = nn.Parameter(torch.zeros(emb_dim))

    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True)
        x_norm = (x - mean) / torch.sqrt(var + self.eps)
        # 引入两个可训练参数 让归一化 时能够相应调整归一化程度 从而一定程度上地保留有效信息
        return self.scale * x_norm + self.shift

    # 激活函数


class GELU(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return 0.5 * x * (1 + torch.tanh(torch.sqrt(torch.tensor(2.0 / torch.pi)) * (x + 0.44715 * torch.pow(x, 3))))


class FeedForward(nn.Module): #前馈神经网络

    #两个线性层 + GELU激活函数
    #线性层的作用 是放大/缩小嵌入层的维度
    #扩大维度 提供中间特征 更加丰富 中间激活层 ----> 连续的线性层 数学上等于一个线性层

    def __init__(self,cfg):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(cfg["emb_dim"],4*cfg["emb_dim"]) ,
            GELU(),
            nn.Linear(4*cfg["emb_dim"], cfg["emb_dim"])
        )

    def forward(self , x):
        return  self.layers(x)

ffn = FeedForward(gpt_config_124M)
x = torch.rand(2,3,768)
out = ffn(x)
print(out.shape)


#快捷链接 残差链接 ----> fix 梯度消失问题 将原始输入 和输出一起作为输出

class ExampleDeepNeuralNetwork(nn.Module):
    def __init__(self , layer_sizes , use_shortcut):
        super().__init__()
        self.use_shortcut = use_shortcut
        self.layers = nn.ModuleList([
            nn.Sequential(nn.Linear(layer_sizes[0],layer_sizes[1]),GELU()),
            nn.Sequential(nn.Linear(layer_sizes[1], layer_sizes[2]), GELU()),
            nn.Sequential(nn.Linear(layer_sizes[3], layer_sizes[4]), GELU()),
            nn.Sequential(nn.Linear(layer_sizes[4], layer_sizes[5]), GELU()),
        ]
        )


# 前馈神经网络模块

# 小型神经网络 由两个线性层和 1个激活函数构成
# 线性层的作用 用于扩大和收缩 嵌入的维度
# 先扩大 后 收拾维度 可以探索更加丰富的表示空间，但是没有数学证明，纯粹的经验说吧
class FeedForward(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(cfg["emb_dim"], 4 * cfg["emb_dim"]),
            GELU(),
            nn.Linear(4 * cfg["emb_dim"], cfg["emb_dim"])
        )

    def forward(self, x):
        return self.layers(x)


class ExampleDeepNeuralNetwork(nn.Module):
    def __init__(self, layer_sizes, use_shortcut):
        super().__init__()
        self.use_shortcut = use_shortcut
        self.layers = nn.ModuleList([
            nn.Sequential(nn.Linear(layer_sizes[0], layer_sizes[1]), GELU()),
            nn.Sequential(nn.Linear(layer_sizes[1], layer_sizes[2]), GELU()),
            nn.Sequential(nn.Linear(layer_sizes[2], layer_sizes[3]), GELU()),
            nn.Sequential(nn.Linear(layer_sizes[3], layer_sizes[4]), GELU()),
            nn.Sequential(nn.Linear(layer_sizes[4], layer_sizes[5]), GELU()),
        ])


    def forward(self, x):
        for layer in self.layers:
            layer_output = layer(x)
            if self.use_shortcut and x.shape == layer_output.shape:
                x = x + layer_output
            else:
                x = layer_output
        return x


layer_sizes = [3,3,3,3,3,1]
sample_input = torch.tensor([[1.,0.,-1.]])
torch.manual_seed(123)
model_without_shortcut = ExampleDeepNeuralNetwork(
    layer_sizes , use_shortcut= False
)

def print_gradients(model,x):
    output = model(x) #前向传播
    target = torch.tensor([[0.]])
    loss = nn.MSELoss()
    loss = loss(output, target)
    loss.backward() #反向传播 torch 自带方法 梯度回传 更新参数

    for name, param in model.named_parameters():
        if 'weight' in name:
            print(f"{name} has gradient mean of {param.grad.abs().mean().item()}")

#打印梯度
print_gradients(model_without_shortcut,sample_input)

## 从 4 ->1 梯度越来越小 反向传播 中的梯度消失
# layers.0.0.weight has gradient mean of 0.00017101920093409717
# layers.1.0.weight has gradient mean of 0.00029431682196445763
# layers.2.0.weight has gradient mean of 0.0009895071852952242
# layers.3.0.weight has gradient mean of 0.0027941337320953608
# 下面添加 shortcut


print('下面添加 shortcut')
torch.manual_seed(123)
model_without_shortcut = ExampleDeepNeuralNetwork(
    layer_sizes , use_shortcut= True
)
print_gradients(model_without_shortcut,sample_input)
print('我是分割线23333')

##
#layers.0.0.weight has gradient mean of 0.006753636058419943
#layers.1.0.weight has gradient mean of 0.005245696287602186
#layers.2.0.weight has gradient mean of 0.008882959373295307
#layers.3.0.weight has gradient mean of 0.04793110489845276
#

