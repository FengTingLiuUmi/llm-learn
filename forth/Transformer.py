import torch
import torch.nn as nn
from torch.linalg import diagonal
from forth import  gpt_learn


class MultiHeadAttention(nn.Module):
    def __init__(self , d_in , d_out ,context_len , dropout , num_heads , qkv_bias = False):
        super().__init__()
        self.d_out =d_out
        self.nums_heads = num_heads
        self.head_dim = d_out // num_heads
        self.W_query = nn.Linear(d_in,d_out,bias=qkv_bias)
        self.W_key = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.out_proj = nn.Linear(d_out,d_out)
        self.dropout =nn.Dropout(dropout)
        self.register_buffer(
            "mask",
            torch.triu(torch.ones(context_len, context_len ), diagonal = 1)
        )

    def forward(self , x):
        batch , num_tokens , d_in = x.shape
        keys = self.W_key(x)
        queries = self.W_query(x)
        values = self.W_value(x)
        keys = keys.view(batch , num_tokens , self.nums_heads , self.head_dim)
        values = values.view(batch, num_tokens, self.nums_heads, self.head_dim)
        queries = queries.view(batch, num_tokens, self.nums_heads, self.head_dim)

        # 1 2 两个维度交换
        keys = keys.transpose(1,2)
        values = values.transpose(1, 2)
        queries = queries.transpose(1, 2)

        #计算点奇

        #掩码部分
        attn_scores = queries @ keys.transpose(2 , 3)
        mask_bool = self.mask.bool()[:num_tokens,:num_tokens] #????
        attn_scores.masked_fill_(mask_bool , -torch.inf) #掩码填充

        #归一化后 分数变为可用的权重
        attn_weights = torch.softmax(attn_scores / keys.shape[-1] ** 0.5, dim=-1)
        attn_weights = self.dropout(attn_weights)

        context_vec = (attn_weights @ values).transpose(1,2)

        context_vec = context_vec.contiguous().view(
            batch , num_tokens , self.d_out
        )
        context_vec = self.out_proj(context_vec) #线性投影
        return  context_vec




class TransformerBlock(nn.Module):
    def __init__(self , cfg):
        super().__init__()
        self.attn = MultiHeadAttention(
            d_in = cfg['emb_dim'],
            d_out= cfg['emb_dim'],
            context_len=cfg['context_length'],
            dropout=cfg['drop_rate'],
            num_heads=cfg['emb_dim'],
            qkv_bias=cfg['qkv_bias']
        )
        #前馈模块
        self.ff = gpt_learn.FeedForward(cfg)
        #线性层1
        self.norm1 = gpt_learn.LayerNorm(cfg['emb_dim'])
        #线性层2
        self.norm2 = gpt_learn.LayerNorm(cfg['emb_dim'])
        self.drop_shortcut = nn.Dropout(cfg['drop_rate'])

    def forward(self , x):
        shortcut = x
        #先经过线性层
        x = self.norm1(x)
        #再过 多头注意力 获得 token之间的信息
        x = self.attn(x)
        #dropout 防止过拟合
        x = self.drop_shortcut(x)
        #补充原始输入 残值链接 避免梯度消失
        x = x + shortcut

        shortcut = x
        #先经过线性层
        x = self.norm2(x)
        #再过前馈网络 根据token间的信息 进一步 优化token自身信息
        x = self.ff(x)
        #dropout
        x = self.drop_shortcut(x)
        #补充原始输入 残值链接 避免梯度消失
        x = x + shortcut
        return x

class GPTModel(nn.Module):
    def __init__(self , cfg):
        super().__init__()
        self.tok_emb = nn.Embedding(cfg['vocab_size'] , cfg['emb_dim'])
        self.pos_emb = nn.Embedding(cfg['context_length'] , cfg['emb_dim'])
        self.drop_emb = nn.Dropout(cfg['drop_rate'])

        # n个transform块
        self.trf_blocks = nn.Sequential(
            *[TransformerBlock(cfg) for _ in range(cfg['n_layers'])]
        )
        #最后再加一个线性输出层
        self.final_norm = gpt_learn.LayerNorm(cfg['emb_dim'])
        #线性输出头 用于将输出的 映射到词汇表 也就是每个词汇输出一个分数
        self.output_head =nn.Linear(
            cfg['emb_dim'] , cfg['vocab_size'] ,bias =False
        )
    def forward(self, in_idx):
        batch_size , seq_len = in_idx.shape
        #计算嵌入表示
        tok_embeds = self.tok_emb(in_idx)
        #计算位置嵌入
        pos_embeds = self.pos_emb(torch.arange(seq_len,device=in_idx.device))
        x = tok_embeds + pos_embeds
        x = self.drop_emb(x)
        x = self.trf_blocks(x)
        x = self.final_norm(x)
        logits = self.output_head(x)
        return logits

def generate_text_sample(model , idx , max_new_tokens , context_size):
    for _ in range(max_new_tokens):
        #截取 防止超出 context_size
        idx_cond = idx[:,-context_size:]
        #关闭梯度计算 推理过程不用反向传播
        with torch.no_grad():
            logits = model(idx_cond)
            #只用最后一个
            logits = logits[:,-1,:]
            probas = torch.softmax(logits , dim= -1)
            #贪心算法  调整通过 温度和 topk topP
            idx_next = torch.argmax(probas , dim= -1 , keepdim= True)
            idx = torch.cat((idx, idx_next) , dim= 1)
    return idx

