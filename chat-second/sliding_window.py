import tiktoken
import torch
from torch.utils.data import Dataset, DataLoader

with open("the-verdict.txt", "r", encoding="utf-8") as verdict_file:
    raw_text = verdict_file.read()

# bpe编码器
tokenizer = tiktoken.get_encoding("gpt2")
# 编码
ids = tokenizer.encode(raw_text)
print(len(ids))
# 定义滑动窗口长度
window_size = 4
for i in range(1, window_size + 1):
    context = ids[:i]
    desired = ids[i]
    print(context, "-->", desired)


# 创建多维数组 为两部分 输入值 和预测值
# 原来py的继承是这样实现的
class GPTDataSetV1(Dataset):
    def __init__(self, txt, tokenizer, max_len, stride):
        self.input_ids = []
        self.target_ids = []
        token_ids = tokenizer.encode(txt)
        for i in range(0, len(token_ids) - max_len, stride):
            input_chunk = token_ids[i:i + max_len]
            target_chunk = token_ids[i + 1:i + max_len + 1]
            self.input_ids.append(torch.tensor(input_chunk))
            self.target_ids.append(torch.tensor(target_chunk))

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return self.input_ids[idx], self.target_ids[idx]


def create_dataloader_v1(text, batch_size=4, max_len=256, stride=128, shuffle=True, drp_last=True, num_workers=0):
    tokenizer1 = tiktoken.get_encoding("gpt2")
    dataset = GPTDataSetV1(text, tokenizer1, max_len, stride)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, drop_last=drp_last,
                            num_workers=num_workers)
    return dataloader


dataloader = create_dataloader_v1(raw_text, batch_size=1, max_len=4, stride=1, shuffle=False)
data_iter = iter(dataloader)
first_batch = next(data_iter)
print(first_batch)
second_batch = next(data_iter)
print(second_batch)

#窗口大小为2 步长为2

dataloader = create_dataloader_v1(raw_text, batch_size=1, max_len=2, stride=2, shuffle=False)
data_iter = iter(dataloader)
first_batch = next(data_iter)
print(first_batch)
second_batch = next(data_iter)
print(second_batch)

#窗口大小为8 步长为2

dataloader = create_dataloader_v1(raw_text, batch_size=1, max_len=8, stride=2, shuffle=False)
data_iter = iter(dataloader)
first_batch = next(data_iter)
print(first_batch)
second_batch = next(data_iter)
print(second_batch)