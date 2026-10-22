import json

import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import tiktoken
from functools import partial



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

    #加载数据集
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
