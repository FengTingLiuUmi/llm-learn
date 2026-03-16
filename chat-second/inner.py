import torch
import sliding_window

input_ids = torch.tensor([2, 3, 4, 5, 1])
vocal_size = 6
output_dim = 3
torch.manual_seed(123)
embedding_layer = torch.nn.Embedding(vocal_size, output_dim)
print(embedding_layer.weight)
print(embedding_layer(torch.tensor([3])))
print(embedding_layer(input_ids))

vocal_size = 50257
output_dim = 256
torch.manual_seed(123)
embedding_layer = torch.nn.Embedding(vocal_size, output_dim)

with open("the-verdict.txt", "r", encoding="utf-8") as verdict_file:
    raw_text = verdict_file.read()

max_len = 4
dataloader = sliding_window.create_dataloader_v1(raw_text, batch_size=8, max_len=max_len, stride=max_len, shuffle=False)
data_iter = iter(dataloader)
inputs, targets = next(data_iter)
print("token Ids:\n", inputs)
print("inputs shape:\n", inputs.shape)

#嵌入
token_embeddings = embedding_layer(inputs)
print(token_embeddings.shape)


context_len = max_len
pos_embedding_layer = torch.nn.Embedding(context_len,output_dim)
pos_embedding = pos_embedding_layer(torch.arange(context_len))
print(pos_embedding.shape)


