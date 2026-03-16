import torch
inputs = torch.tensor(
    [[0.43,0.15,0.89],
    [0.55,0.87,0.66],
    [0.57,0.85,0.64],
    [0.22,0.58,0.33],
    [0.77,0.25,0.10],
    [0.05,0.80,0.55]]
)
query = inputs[1]
attn_score_2 = torch.empty(inputs.shape[0])
for i,x_i in enumerate(inputs):
    attn_score_2[i] = torch.dot(x_i,query)
print(attn_score_2)

#归一化 说白了就是转化为总和为1情况 一个是减少总量级 减少梯度损失 一个是减少方便进行概率解释
attn_score_weight_2 = attn_score_2 / attn_score_2.sum()
print(attn_score_weight_2)
print(attn_score_weight_2.sum())

#在实际中一般使用softmax 进行归一化 因为他在梯度损失更小  softmax 也可以保障总是正值 todo 理解下softmax
def softmax_naive(x):
    return torch.exp(x)/torch.exp(x).sum(dim=0)
attn_weight_2_naive = softmax_naive(attn_score_2)
print("attetion weights",attn_weight_2_naive)
print("Sum:",attn_weight_2_naive.sum())

query = inputs[1]
context_vec_2 = torch.zeros(query.shape)
for i,x_i in enumerate(inputs):
    context_vec_2 += attn_weight_2_naive[i] * x_i
print(context_vec_2)


#计算所有的 上下文 通过点积 计算所有的注意力权重
attn_scores = torch.empty(6,6)
for i,x_i in enumerate(inputs):
    for j , x_j in enumerate(inputs):
        attn_scores[i,j] = torch.dot(x_i,x_j)
print(attn_scores)

#使用矩阵乘法代替 cpu串行 -> gpu并行
new_attn_score = inputs @ inputs.T
print(new_attn_score)

#对权重进行归一化处理 得到最终的权重
attn_weight = torch.softmax(new_attn_score,dim=-1)
print(attn_weight)

#权重 * 每一行得到对应的上下文向量
all_context_vecs =  attn_weight @ inputs
print(all_context_vecs)


#可训练权重
#第二个词的 词嵌入
x_2 = inputs[1]
d_in = inputs.shape[1]
d_out = 2

#初始化 自适应权重矩阵
torch.manual_seed(123)
W_query = torch.nn.Parameter(torch.rand(d_in,d_out),requires_grad=False)
W_key = torch.nn.Parameter(torch.rand(d_in,d_out),requires_grad=False)
W_value = torch.nn.Parameter(torch.rand(d_in,d_out),requires_grad=False)

#计算 词嵌入 经过每个权重矩阵后的值
query_2 = x_2 @ W_query
key_2 = x_2 @ W_key
value_2 = x_2 @ W_value
print(query_2)

#整个 句子的词嵌入的权重计算
keys = inputs @ W_key
values = inputs @ W_value
print("keys.shape",keys.shape)
print("value.shape",values.shape)

#计算最终的注意力分数
keys_2 = keys[1]
attn_score_22 = query_2.dot(keys_2)
print(attn_score_22)

#整个的注意力分数
attn_score_2 = query_2 @ keys.T
print(attn_score_2)
