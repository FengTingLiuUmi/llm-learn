import sys
from pathlib import Path
import json

import torch
import re
import time
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import tiktoken
from functools import partial
import gpt_download
from forth.Transformer import GPTModel,generate_text_sample
from after.data_prepare import  txt_2_token_ids ,token_ids_2_txt

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

def generate_and_print_sample(model , tokenizer ,device, start_context):
    model.eval()
    context_size = model.pos_emb.weight.shape[0]
    encoded = txt_2_token_ids(start_context , tokenizer).to(device)
    with torch.no_grad():
        token_ids = generate_text_sample(model = model , idx = encoded , max_new_tokens= 50
                                                           , context_size= context_size)
        decoded_text = token_ids_2_txt(token_ids , tokenizer)
        print(decoded_text.replace('\n' , ' '))
        model.train()

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



def load_from_file(file_path):
    with open(file_path,'r',encoding='utf-8') as file:
        return json.load(file)

def split_data(data):
    train_portion = int(len(data) * 0.85)
    test_portion = int(len(data) * 0.1)
    val_portion = len(data) - test_portion - train_portion

    train_data = data[:train_portion]
    test_data = data[train_portion:test_portion+train_portion]
    val_data = data[train_portion + test_portion :]
    return train_data , test_data ,val_data

#格式化提示词
def format_input(entry):
    instruction_text = (
        f"Below is an instruction that describes a task. "
        f"Write a response that appropriately completes the request."
        f"\n\n### Instruction:\n{entry['instruction']}"
    )

    input_text = (
        f"\n\n### Input:\n{entry['input']}" if entry["input"] else ""
    )
    return instruction_text + input_text

class InstructionDataset(Dataset):
    def __init__(self , data , tokenizer):
        self.data =data
        self.encoded_texts = []
        for entry in data:
            instruction_plus_input = format_input(entry)
            response_text = f"\n\n### Response:\n{entry['output']}"
            full_text = instruction_plus_input + response_text
            self.encoded_texts.append(
                tokenizer.encode(full_text)
            )
    def __getitem__(self, index):
        return self.encoded_texts[index]

    def __len__(self):
        return len(self.data)

def custom_collate_draft_1(
        batch ,
        pad_token_id = 50256 ,
        device = 'cpu'
):
    #批次中最长的
    batch_max_len = max(len(item) +1 for item in batch)
    inputs_list = []
    for item in batch:
        new_item = item.copy()
        #填充结束符号
        new_item += [pad_token_id]
        padded = (
            new_item + [pad_token_id] * (batch_max_len - len(new_item))
        )
        inputs = torch.tensor(padded[:-1])
        inputs_list.append(inputs)
    return torch.stack(inputs_list).to(device)

#计算目标词元Id
def custom_collate_draft_2(
        batch ,
        pad_token_id = 50256 ,
        device = 'cpu'
):
    #批次中最长的
    batch_max_len = max(len(item + 1) for item in batch)
    inputs_list , target_list = [] , []
    for item in batch:
        new_item = item.copy()
        #填充结束符号
        new_item += [pad_token_id]
        padded = (
            new_item + [pad_token_id] * (batch_max_len - len(new_item))
        )
        inputs = torch.tensor(padded[:-1])
        targets = torch.tensor(padded[1:]) #左移得到目标
        inputs_list.append(inputs)
        target_list.append(targets)
    return torch.stack(inputs_list).to(device) , torch.stack(target_list).to(device)

#计算目标词元Id
def custom_collate_fn(
        batch ,
        pad_token_id = 50256 ,
        ignore_index = -100,
        allowed_max_len = None,
        device = 'cpu'
):
    #批次中最长的
    batch_max_len = max(len(item) + 1 for item in batch)
    inputs_list , target_list = [] , []
    for item in batch:
        new_item = item.copy()
        #填充结束符号
        new_item += [pad_token_id]
        padded = (
            new_item + [pad_token_id] * (batch_max_len - len(new_item))
        )
        inputs = torch.tensor(padded[:-1])
        targets = torch.tensor(padded[1:]) #左移得到目标

        #设置为ignore
        mask = targets == pad_token_id
        indices = torch.nonzero(mask).squeeze()
        if indices.numel() > 1:
            targets[indices[1:]] = ignore_index
        if allowed_max_len is not None: #截断
            inputs = inputs[:allowed_max_len]
            targets = targets[:allowed_max_len]

        inputs_list.append(inputs)
        target_list.append(targets)
    return torch.stack(inputs_list).to(device) , torch.stack(target_list).to(device)

if __name__ == "__main__":

    #加载数据集/mnt/f/pyProject/llm-learn-new/final/instruction-data.json
    data = load_from_file('instruction-data.json')

    tokenizer = tiktoken.get_encoding("gpt2")
    print(tokenizer.encode("<|endoftext|>", allowed_special={"<|endoftext|>"}))
    inputs_1 = [0, 1, 2, 3, 4]
    inputs_2 = [5, 6]
    inputs_3 = [7, 8, 9]
    batch = (
        inputs_1,
        inputs_2,
        inputs_3
    )
    print(custom_collate_draft_1(batch))
    inputs, targets = custom_collate_fn(batch)
    print(inputs)
    print(targets)
#-100默认在计算交叉商损失是会被忽略  可以考虑 掩码指令部分 让模型专注与结果部分，而不是过拟合 延续指令

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('device: ',device)
    customized_collate_fn = partial(
        custom_collate_fn,
        device=device,
        allowed_max_len=1024
    )

    #初始化数据加载器

    num_workers = 0
    batch_size = 8

    torch.manual_seed(123)

    train_data , test_data,val_data = split_data(data)

    train_dataset =InstructionDataset(train_data , tokenizer)

    train_loader = DataLoader(
        train_dataset ,
        batch_size = batch_size,
        collate_fn=customized_collate_fn,
        shuffle=True,
        drop_last=True,
        num_workers=num_workers
    )

    test_dataset =InstructionDataset(test_data , tokenizer)

    test_loader = DataLoader(
        test_dataset ,
        batch_size = batch_size,
        collate_fn=customized_collate_fn,
        shuffle=False,
        drop_last=False,
        num_workers=num_workers
    )

    val_dataset = InstructionDataset(val_data, tokenizer)

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        collate_fn=customized_collate_fn,
        shuffle=False,
        drop_last=False,
        num_workers=num_workers
    )

    print("Train loader:")
    for inputs, targets in train_loader:
        print(inputs.shape, targets.shape)


#如果 模型参数过小 在支持指令微调这种复杂任务 数学证明有么???


    #加载mid gpt2模型
    BASE_CONFIG = {
        "vocab_size": 50257,  # 词汇表大小
        "context_length": 1024,  # 上下文长度
        "drop_rate": 0.0,  # dropout 率
        "qkv_bias": True  # 查询-键-值偏置
    }

    model_configs = {
        "gpt2-small (124M)": {"emb_dim": 768, "n_layers": 12, "n_heads": 12},
        "gpt2-medium (355M)": {"emb_dim": 1024, "n_layers": 24, "n_heads": 16},
        "gpt2-large (774M)": {"emb_dim": 1280, "n_layers": 36, "n_heads": 20},
        "gpt2-xl (1558M)": {"emb_dim": 1600, "n_layers": 48, "n_heads": 25},
    }

    CHOOSE_MODEL = "gpt2-medium (355M)"
    BASE_CONFIG.update(model_configs[CHOOSE_MODEL])

    model_size = CHOOSE_MODEL.split(" ")[-1].lstrip("(").rstrip(")")

    settings, params = gpt_download.download_and_load_gpt2(
        model_size=model_size,
        models_dir="gpt2"
    )

    model = GPTModel(BASE_CONFIG)
    gpt_download.load_weights_into_gpt(model, params)
    model.eval()

    torch.manual_seed(123)
    input_text = format_input(val_data[0])
    print(input_text)

    token_ids = generate(
        model=model,
        idx=txt_2_token_ids(input_text, tokenizer),
        max_new_tokens=35,
        context_size=BASE_CONFIG["context_length"],
        eos_id=50256,
    )
    generated_text = token_ids_2_txt(token_ids, tokenizer)



    response_text = generated_text[len(input_text):].strip()
    print('input_text:' , input_text)
    print('response : ', response_text)

    #微调模型
    model.to(device)
    torch.manual_seed(123)

    with torch.no_grad():
        train_loss = cal_loss_loader(
            train_loader, model, device, num_batch=5
        )
        val_loss = cal_loss_loader(
            val_loader, model, device, num_batch=5
        )

    print("Training loss:", train_loss)
    print("Validation loss:", val_loss)

    #进行指令微调 这里因为显卡是7900xt 所以这里只能在wsl下的rocm跑
    print("now:", train_loss)
    start_time = time.time()
    torch.manual_seed(123)
    optimizer = torch.optim.AdamW(
    model.parameters(), lr=0.00005, weight_decay=0.1
    )
    num_epochs = 2

    train_losses, val_losses, tokens_seen = train_model_simple(
    model, train_loader, val_loader, optimizer, device,
    num_epochs=num_epochs, eval_freq=5, eval_iter=5,
    start_context=format_input(val_data[0]), tokenizer=tokenizer
    )

    end_time = time.time()
    execution_time_minutes = (end_time - start_time) / 60
    print(f"Training completed in {execution_time_minutes:.2f} minutes.")

    # 用cpu 跑训练 Ep 2 (Step 000230 )Train loss 0.716val loss 1.198
    # Below is an instruction that describes a task. Write a response that appropriately completes the request.  ### Instruction: Convert the active sentence to passive: 'The chef cooks the meal every day.'  ### Response: The meal is typically between 5 minutes.<|endoftext|>The meal is typically between 5.<|endoftext|>The meal.<|endoftext|>The meal is typically between 5.<|endoftext|>The meal is typically between 5.<|endoftext|>The meal.<|endoftext|>The meal is
    # Training completed in 55.46 min

    file_name = f"{re.sub(r'[ ()]', '',CHOOSE_MODEL)} -sft.pth"
    torch.save(model.state_dict(),file_name)
    print('saved as', file_name)