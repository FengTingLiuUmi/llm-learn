import torch


import forth.Transformer
from forth import Transformer
import tiktoken
from chat_second import sliding_window

gpt_cfg_124M = {
    "vocab_size": 50257,  # 词汇表大小
    "context_length": 256,  # 上下文长度
    "emb_dim": 768,  # 嵌入维度 就是 表示一个词有多少 信息维度
    "n_heads": 12,  # 注意力头的数量
    "n_layers": 12,  # 层数
    "drop_rate": 0.1,  # drop_out率
    "qkv_bias": False  # 偏置
}


def txt_2_token_ids(text, tokenizer):
    encoded = tokenizer.encode(text, allowed_special={'<|endoftext|>'})
    encoded_tensor = torch.tensor(encoded).unsqueeze(0)
    return encoded_tensor


def token_ids_2_txt(token_ids, tokenizer):
    return tokenizer.decode(token_ids.squeeze(0).tolist())


inputs = torch.tensor([[16833, 3626, 6100],  # ["every effort moves",
                       [40, 1107, 588]])  # "I really like"]

targets = torch.tensor([[3626, 6100, 345],  # [" effort moves you",
                        [1107, 588, 11311]])  # " really like chocolate"]

model = Transformer.GPTModel(gpt_cfg_124M)

with torch.no_grad():
    logits = model(inputs)
probas = torch.softmax(logits, dim=-1)

token_ids = torch.argmax(probas, dim=-1, keepdim=True)
print("Token IDs:\n", token_ids)

text_idx = 0
target_probas_1 = probas[text_idx, [0, 1, 2], targets[text_idx]]
print("Text 1:", target_probas_1)

text_idx = 1
target_probas_2 = probas[text_idx, [0, 1, 2], targets[text_idx]]
print("Text 2:", target_probas_2)

log_probas = torch.log(torch.cat((target_probas_1, target_probas_2)))
print(log_probas)

avg_log_probas = torch.mean(log_probas)
print(avg_log_probas)

neg_avg_log_probas = avg_log_probas * -1
print(neg_avg_log_probas)  # 交叉商损失 -----> pytorch.cross_entropy

# 1. 原始得分
# 2. 归一化为 综合为1的概率分布
# 3. 目标概率为多少?
# 4. 对目标概率 取对数 会变为负值 将概率连乘变为加法
# 5. 平均对数概率
# 转为非负数 ln(1) = 0 即准确时为0 不准时特别大 可以作为损失函数

# 对于离散结果进行度量 如标签

logits_flat = logits.flatten(0, 1)
targets_flat = targets.flatten()
#
loss = torch.nn.functional.cross_entropy(logits_flat, targets_flat)
print(loss)

file_path = '../chat_second/the-verdict.txt'
with open(file_path, 'r', encoding='utf-8') as file:
    text_data = file.read()
tokenizers = tokenizer = tiktoken.get_encoding("gpt2")

# 拆分 训练集和测试集
train_ratio = 0.90
split_idx = int(train_ratio * len(text_data))
train_data = text_data[:split_idx]
val_data = text_data[split_idx:]

torch.manual_seed(123)
train_loader = sliding_window.create_dataloader_v1(train_data, batch_size=2,
                                    max_len=gpt_cfg_124M['context_length'],
                                    stride=gpt_cfg_124M['context_length'],
                                    drp_last=True,
                                    shuffle=True,
                                    num_workers=0)

val_loader = sliding_window.create_dataloader_v1(val_data, batch_size=2,
                                    max_len=gpt_cfg_124M['context_length'],
                                    stride=gpt_cfg_124M['context_length'],
                                    drp_last=False,
                                    shuffle=False,
                                    num_workers=0)

#计算对应批次的交叉商损失
def cal_loss_batch(input_batch , target_batch , model ,device):
    input_batch = input_batch.to(device) #移动到gpu上
    target_batch = target_batch.to(device)
    logits = model(input_batch)
    loss = torch.nn.functional.cross_entropy(logits.flatten(0,1),target_batch.flatten())
    return loss

def cal_loss_loader(data_loader , model , device , num_batch = None):
    total_loss = 0.
    if len(data_loader) == 0:
        return float('nan')
    elif num_batch is None:
        num_batch = len(data_loader)
    else:
        num_batch = min(num_batch , len(data_loader))
    for i , (input_batch , target_batch) in enumerate(data_loader):
        if i < num_batch:
            loss = cal_loss_batch(input_batch , target_batch , model ,device)
            total_loss += loss.item()
        else:
            break
    return total_loss / num_batch

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)
with torch.no_grad():
    train_loss = cal_loss_loader(train_loader, model ,device)
    val_loss = cal_loss_loader(val_loader, model, device)
print('train loss is' , train_loss)
print('validation loss is', val_loss)

#1. 遍历训练轮次
#遍历批次
# 重置损失梯度
#计算损失
#反向传播 计算 损失梯度
#根据损失梯度更新模型权重
#打印
#可视化

def train_model_simple(model , train_loader , val_loader , optimizer ,device ,num_epochs ,
                       eval_freq , eval_iter , start_context , tokenizer):
    train_losses , val_losses , track_token_seen = [] , [] , []
    tokens_seen , global_step = 0 , -1
    for epoch in range(num_epochs):
        model.train()
        for input_batch , target_batch in train_loader:
            optimizer.zero_grad() #重置损失梯度
            loss = cal_loss_batch(
                input_batch , target_batch , model ,device
            )
            loss.backward()
            optimizer.step()
            tokens_seen += input_batch.numel()
            global_step += 1

            if global_step % eval_freq == 0 :
                train_loss , val_loss = evaluate_model(
                    model , train_loader , val_loader , device , eval_iter
                )
                train_losses.append(train_loss)
                val_losses.append(val_loss)
                track_token_seen.append(tokens_seen)
                print(f"Ep {epoch + 1} (Step {global_step:06d} )"
                      f"Train loss {train_loss:.3f}"
                      f"val loss {val_loss:.3f}"
                      )

        generate_and_print_sample(model , tokenizer , device ,start_context)
    return train_losses , val_losses , track_token_seen

def evaluate_model(model , train_loader , val_loader , device , eval_iter):
    model.eval()
    with torch.no_grad():
        train_loss = cal_loss_loader(
            train_loader , model, device, num_batch= eval_iter
        )
        val_loss = cal_loss_loader(
            val_loader , model, device, num_batch= eval_iter
        )
    model.train()
    return train_loss , val_loss

def generate_and_print_sample(model , tokenizer ,device, start_context):
    model.eval()
    context_size = model.pos_emb.weight.shape[0]
    encoded = txt_2_token_ids(start_context , tokenizer).to(device)
    with torch.no_grad():
        token_ids = forth.Transformer.generate_text_sample(model = model , idx = encoded , max_new_tokens= 50
                                                           , context_size= context_size)
        decoded_text = token_ids_2_txt(token_ids , tokenizers)
        print(decoded_text.replace('\n' , ' '))
        model.train()


torch.manual_seed(123)
model = forth.Transformer.GPTModel(gpt_cfg_124M)
model.to(device)
optimizer = torch.optim.AdamW(
    model.parameters() ,
    lr = 0.0004 , weight_decay= 0.1
)
num_epochs = 10
train_losses , val_losses , token_seen = train_model_simple(
    model , train_loader , val_loader , optimizer , device ,
    num_epochs = num_epochs , eval_freq= 5 , eval_iter= 5 ,
    start_context='Every effort moves you' , tokenizer=tokenizer
)
#Ep 10 (Step 000085 )Train loss 0.551val loss 6.332
#Every effort moves you?"  "Yes--quite insensible to the irony. She wanted him vindicated--and by me!"  He laughed again, and threw back his head to look up at the sketch of the donkey. "There were days when I


def generate(model , idx , max_new_tokens , context_size ,
             temperature = 0.0 , top_k = None , eos_id =None):
    for _ in range(max_new_tokens):
        idx_cond=idx[:,-context_size:]
        with torch.no_grad():
            logits = model(idx_cond)
        logits = logits[:,-1,:]
        if top_k is not None:
            top_logits , _ = torch.topk(logits,top_k)
            min_val = top_logits[:,-1]
            logits = torch.where(
                logits < min_val ,
                torch.tensor(float('-inf')).to(logits.device),
                logits
            )
        if temperature > 0.0:
            logits = logits / temperature
            probs = torch.softmax(logits , dim = -1)
            idx_next = torch.multinomial(probs , num_samples= 1)
        else:
            idx_next = torch.argmax(logits , dim = -1 , keepdim= True)
        if idx_next == eos_id: #结束的词元id 停止生成
            break
        idx = torch.cat((idx,idx_next) , dim= -1)
    return idx

torch.manual_seed(123)
token_ids = generate(
    model=model,
    idx=txt_2_token_ids("Every effort moves you", tokenizer),
    max_new_tokens=15,
    context_size=gpt_cfg_124M["context_length"],
    top_k=25,
    temperature=1.4
)
print("Output text:\n", token_ids_2_txt(token_ids, tokenizer))


#保存参数
torch.save(model.state_dict() , 'model.pth')

model = forth.Transformer.GPTModel(gpt_cfg_124M)
model.load_state_dict(torch.load('model.pth' , map_location = device))
model.eval()