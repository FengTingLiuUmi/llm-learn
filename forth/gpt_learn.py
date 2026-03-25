import torch
import torch.nn as nn

gpt_config_124M = {
    "vocab_size": 50257,  # 词汇表大小
    "context_length": 1024,  # 上下文长度
    "emb_dim": 768,  # 嵌入味道
    "n_heads": 12,  # 注意力头的数量
    "n_layers": 12,  # 层数
    "drop_rate": 0.1,  # drop_out率
    "qkv_bias": False  # 偏置
}


class DummpyGPTModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        #token层
        self.tok_emb = nn.Embedding(cfg["vocab_size"], cfg["emb_dim"])
        #token嵌入层
        self.pos_emb = nn.Embedding(cfg["context_len"], cfg["emb_dim"])
        #dropout
        self.drop_emb = nn.Dropout(cfg["drop_rate"])
        #transformer块
        self.trf_blocks = nn.Sequential(
            *[DummyTransformerBlock(cfg)
              for _ in range(cfg["n_layers"])]
        )
        #归一化层
        self.final_norm = DummyLayerNorm(cfg["emb_dim"])
        #线性输出层
        self.out_head = nn.Linear(
            cfg["emb_dim"], cfg["vocab_size"], bias=False
        )
    def forward(self , in_idx):
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

        def forward(shelf, x):
            return x


class DummyLayerNorm(nn.Module):
    def __init__(self, normalized_shape, eps=1e-5):
        super().__init__()

    def forward(self, x):
        return x
