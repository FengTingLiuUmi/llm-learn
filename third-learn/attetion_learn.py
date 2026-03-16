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
