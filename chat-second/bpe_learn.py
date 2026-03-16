from importlib.metadata import version
import tiktoken

#bpe学习
tokenizer = tiktoken.get_encoding("gpt2")
base_text = "Akwirw ier."
ids = tokenizer.encode(base_text)
print(ids)
res = tokenizer.decode(ids)
print(res)