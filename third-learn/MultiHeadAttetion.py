import torch
import torch.nn as nn


class MultiHeadAttention(nn.Module):
    """
    多头注意力机制 (Multi-Head Attention)

    核心思想：将输入通过多个并行的注意力"头"处理，每个头学习不同的注意力模式，
    最后将所有头的输出合并。这样可以让模型同时关注不同位置的不同表示子空间。

    计算流程：
    1. 输入 x 经过三个线性层，得到 Query、Key、Value
    2. 将 Q、K、V 拆分成多个头
    3. 每个头独立计算注意力分数（缩放点积注意力）
    4. 应用因果掩码（确保只能看到当前位置及之前的内容）
    5. 加权求和得到每个头的输出
    6. 拼接所有头的输出，再通过一个线性层输出最终结果
    """

    def __init__(self, d_in, d_out, context_len, dropout, num_heads, qkv_bias=False):
        """
        初始化多头注意力层

        参数说明：
        - d_in:      输入特征维度
        - d_out:     输出特征维度（也是 Q、K、V 的投影维度）
        - context_len: 上下文长度（序列最大长度，用于生成因果掩码）
        - dropout:   Dropout 概率，用于防止过拟合 丢弃
        - num_heads: 注意力头的数量
        - qkv_bias:  是否在 Q、K、V 的线性投影中使用偏置项
        """
        super().__init__()

        # ========== 维度校验 ==========
        # d_out 必须能被 num_heads 整除，这样才能均匀分配给每个头 ----》需要将 输出 按照头数拆分 因为这里是将多头矩阵合并为了一个矩阵
        # 例如：d_out=64, num_heads=8 → 每个头的维度 head_dim=8
        assert (d_out % num_heads == 0), \
            "d_out must be divisible by num_heads"

        self.d_out = d_out
        self.num_heads = num_heads
        self.head_dim = d_out // num_heads  # 每个头的维度

        # ========== Q、K、V 投影层 ==========
        # 将输入 x 投影到 Query、Key、Value 空间
        # 这里的投影维度都是 d_out，之后会被拆分成 num_heads 个头

        #定义一个神经网络的 线性层 y = ax + b 的线性变化 b就是偏置
        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)  # Query 投影
        self.W_key = nn.Linear(d_in, d_out, bias=qkv_bias)    # Key 投影
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)  # Value 投影

        # ========== 输出投影层 ==========
        # 将所有头的输出拼接后，再通过一个线性层进行融合
        # 这是多头注意力的关键：让不同的头可以交互、融合信息

        #这一层 相当于 给多头加上了权重 来表示哪个头更加重要
        self.out_proj = nn.Linear(d_out, d_out)

        # ========== Dropout 层 ==========
        # 对注意力权重进行 dropout，增加模型鲁棒性
        self.dropout = nn.Dropout(dropout)

        # ========== 因果掩码 (Causal Mask) ==========
        # 创建上三角矩阵（对角线上方为1，下方为0）
        # 用于实现因果注意力：位置 i 只能看到位置 0~i 的信息
        #
        # 示例（context_len=4）：
        # mask = [[0, 1, 1, 1],   # 位置0只能看到位置0
        #         [0, 0, 1, 1],   # 位置1能看到位置0,1
        #         [0, 0, 0, 1],   # 位置2能看到位置0,1,2
        #         [0, 0, 0, 0]]   # 位置3能看到位置0,1,2,3
        #
        # register_buffer: 将张量注册为 buffer，它会随模型移动到 GPU/CPU，
        # 但不会被当作模型参数进行梯度更新

        #全1矩阵 -> 上三角 ->
        self.register_buffer("mask", torch.triu(torch.ones(context_len, context_len), diagonal=1))

    def forward(self, x):
        """
        前向传播

        输入形状：(batch_size, num_tokens, d_in)
        输出形状：(batch_size, num_tokens, d_out)

        参数：
        - x: 输入张量，形状为 (b, num_tokens, d_in)
        """
        b, num_tokens, d_in = x.shape  # b=批次大小, num_tokens=序列长度

        # ========== 步骤1: 线性投影 ==========
        # 将输入投影到 Q、K、V 空间
        # 形状变化: (b, num_tokens, d_in) → (b, num_tokens, d_out)
        keys = self.W_key(x)
        queries = self.W_query(x)
        values = self.W_value(x)

        # ========== 步骤2: 拆分成多个头 ==========
        # 将 d_out 维度拆分成 (num_heads, head_dim)
        # 形状变化: (b, num_tokens, d_out) → (b, num_tokens, num_heads, head_dim)
        #
        # 为什么要拆分？这样可以让不同的头独立计算注意力，学习不同的模式

        #将原本的 d_out 转化为 num_heads ，head_dim
        keys = keys.view(b, num_tokens, self.num_heads, self.head_dim)
        queries = queries.view(b, num_tokens, self.num_heads, self.head_dim)
        values = values.view(b, num_tokens, self.num_heads, self.head_dim)

        # ========== 步骤3: 转置，使头维度在前 ==========
        # 形状变化: (b, num_tokens, num_heads, head_dim) → (b, num_heads, num_tokens, head_dim)
        #
        # 为什么要转置？为了方便后续的矩阵运算：
        # - 现在每个头可以独立处理 (num_tokens, head_dim) 的数据
        # - 可以同时计算所有头的注意力，而不需要循环

        #高维度张量的乘法 还是乘最后的两维度 所以讲头放到前面 使得能够正常地乘法
        keys = keys.transpose(1, 2)
        queries = queries.transpose(1, 2)
        values = values.transpose(1, 2)

        #矩阵乘法 消耗大

        # ========== 步骤4: 计算注意力分数 ==========
        # 缩放点积注意力: Attention(Q, K, V) = softmax(QK^T / √d_k) * V
        #
        # queries @ keys.transpose(1, 2) 计算 Q 和 K 的点积
        # 形状: (b, num_heads, num_tokens, head_dim) @ (b, num_heads, head_dim, num_tokens)
        #     = (b, num_heads, num_tokens, num_tokens)
        #
        # attn_scores[i, h, j, k] 表示第 i 个样本、第 h 个头、位置 j 对位置 k 的注意力分数
        attn_scores = queries @ keys.transpose(1, 2)

        # ========== 步骤5: 应用因果掩码 ==========
        # 将掩码矩阵中值为 True 的位置填充为负无穷
        # 这样 softmax 后这些位置的权重会变成 0
        #
        # mask_bool 形状: (num_tokens, num_tokens)
        # 我们只取当前序列长度对应的掩码部分
        mask_bool = self.mask.bool()[:num_tokens, :num_tokens]

        # masked_fill_ 是原地操作，将 mask_bool 为 True 的位置填充为 -inf
        # 这样在 softmax 时，这些位置的权重会变成 0（因为 e^(-inf) = 0）
        attn_scores.masked_fill_(mask_bool, -torch.inf)

        # ========== 步骤6: Softmax 归一化 ==========
        # 将注意力分数归一化为概率分布（每行和为1）
        #
        # 除以 √head_dim 进行缩放，防止点积值过大导致 softmax 梯度消失
        # 原因：当维度较大时，点积的方差会增大，softmax 会趋向于 one-hot
        attn_weights = torch.softmax(attn_scores / keys.shape[-1] ** 0.5, dim=-1)

        # ========== 步骤7: Dropout ==========
        # 对注意力权重应用 dropout，训练时随机丢弃一些连接
        attn_weights = self.dropout(attn_weights)

        # ========== 步骤8: 加权求和 ==========
        # 用注意力权重对 values 进行加权求和
        # 形状: (b, num_heads, num_tokens, num_tokens) @ (b, num_heads, num_tokens, head_dim)
        #     = (b, num_heads, num_tokens, head_dim)
        #
        # 然后转置回 (b, num_tokens, num_heads, head_dim)，准备拼接
        context_vec = (attn_weights @ values).transpose(1, 2)

        # ========== 步骤9: 拼接所有头 ==========
        # 将所有头的输出拼接成一个向量
        # 形状: (b, num_tokens, num_heads, head_dim) → (b, num_tokens, d_out)
        #
        # contiguous(): 确保张量在内存中是连续存储的（view 操作要求）
        # 这是因为 transpose 后张量可能不再连续
        context_vec = context_vec.contiguous().view(b, num_tokens, self.d_out)

        # ========== 步骤10: 输出投影 ==========
        # 通过线性层融合所有头的信息
        # 这是多头注意力的关键：让不同的头可以交互
        context_vec = self.out_proj(context_vec)

        return context_vec



