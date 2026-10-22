import urllib.request
import re

# 读取文本 作为样本
with open("the-verdict.txt", "r", encoding="utf-8") as verdict_file:
    raw_text = verdict_file.read()
    print("total char count of text is", len(raw_text))
    print(raw_text[:99])
# 分词示例1 类别java的分词 split 基于空白分词
result1 = re.split(r'(\s)', raw_text)
print(result1)
# 分词示例1 缺陷 无法区分标点 蕴含标点 因此 额外进行删除标点
# 分词示例 2 在空白和标点处分词
result2 = re.split(r'([,.]|\s)', raw_text)
#去除空白 如果是强依赖空白的场景 就不要区分 比如python代码
result2 = [i for i in result2 if i.strip()]
print(result2)
#3完善下 区分大部分标点
result3 = re.split(r'([,.:;?_!"()\']|--|\s)', raw_text)
result3 = [i for i in result3 if i.strip()]
print(result3)
#上面的result 就是分割出的token了

#映射创建出词汇表

#补充特殊词元 未知 和 结束
result3.extend(["<|endoftext|>","<|unk|>"])

token_map = {token:i for i,token in enumerate(result3)}
for i,item in enumerate(token_map.items()):
    if i > 0:
        break
    print(item)

#定义的分词器类
class SimpleTokenizerV1:
    def __init__(self,token_map):
        self.str2int = token_map
        self.int2str = {i:s for s,i in token_map.items()}
    #预计算 将文本进行分词
    def preprocess(self,text):
        token_list = re.split(r'([,.:;?_!"()\']|--|\s)', text)
        token_list = [i for i in token_list if i.strip()]
        return  token_list
    #编码 将文本转成 tokenId集合
    def encode(self,text):
        token_list = self.preprocess(text)
        return [self.str2int[s] for s in token_list]
    #解码 tokenId转回词元
    def decode(self,ids):
        text = " ".join(self.int2str[i] for i in ids)
        text = re.sub(r'\s+([,.:;?_!"()\'])',r'\1',text)
        return text
#定义的分词器类2
class SimpleTokenizerV2:
    def __init__(self,token_map):
        self.str2int = token_map
        self.int2str = {i:s for s,i in token_map.items()}
    #预计算 将文本进行分词
    def preprocess(self,text):
        token_list = re.split(r'([,.:;?_!"()\']|--|\s)', text)
        token_list = [i for i in token_list if i.strip()]
        return  token_list
    #编码 将文本转成 tokenId集合
    def encode(self,text):
        token_list = self.preprocess(text)
        #将未知词元转化为unk
        token_list = [token if token in self.str2int else "<|unk|>" for token in token_list]
        return [self.str2int[s] for s in token_list]
    #解码 tokenId转回词元
    def decode(self,ids):
        text = " ".join(self.int2str[i] for i in ids)
        text = re.sub(r'\s+([,.:;?_!"()\'])',r'\1',text)
        return text



#模拟分词器的编码 解码能力
tokenizer = SimpleTokenizerV1(token_map)
this_is_a_text = "you know who"
ids = tokenizer.encode(this_is_a_text)
print(ids)
print(tokenizer.decode(ids))

#改造后的分词器
tokenizer2 = SimpleTokenizerV2(token_map)
get_a_son_text = "you are my son."
response_text = "you are my father!"
ids = tokenizer2.encode(" <|endoftext|> ".join((get_a_son_text,response_text)))
print(ids)
print(tokenizer2.decode(ids))
