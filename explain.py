from tokenizers import Tokenizer
from main import *
from torch.nn import functional as F

WORD_NAME = "20B_tokenizer.json"
tokenizer = Tokenizer.from_file(WORD_NAME)
ids = tokenizer.encode('你杀了我吧呜呜呜').ids
out, state = model.forward(ids, state = None)


# NOTE: Sample
# out: (50277) # 全词汇, 说明是下一个token的可能性
logits = out
probs = F.softmax(logits.float(), dim=-1) # 获取分布
sorted_ids = torch.argsort(probs) # 可能性从低到高排序, 获取下标
sorted_probs = probs[sorted_ids] # 对应的可能性从低到高排序
sorted_probs = torch.flip(sorted_probs, dims=(0,))
# 这个没懂，像是非伯纳齐数列一样堆叠起来，但是为什么要转成cpu?
cumulative_probs = torch.cumsum(sorted_probs, dim=-1).cpu().numpy()
true_falses = cumulative_probs > top_p # 50277个正负, 19之后就是True了
first_true_idx = np.argmax(true_falses) # 18，第一个出现True的下标，不知道这样的算法有什么必要
cutoff = float(sorted_probs[np.argmax(cumulative_probs > top_p)]) # 获取了下标18的可能性?
# 将所有可能性小于0.0041的都置零了
probs[probs < cutoff] = 0
# 默认top_k是100
dd =  sorted_ids[:-top_k]
if top_k < len(probs) and top_k > 0:
    probs[sorted_ids[:-top_k]] = 0 # 把从低到高排名50177之内的可能性都置0了
if temperature != 1.0:
    # 如果温度设置0.5, prob变成平方
    # 如果温度设置2, prob变成开平方根
    # 因为prob小于1, 2相当于抬高可能性的曲线（特别是山脚下的地方）
    # 因为prob小于1, 0.5相当于压低可能性的曲线（特别是靠近山脚下的地方）
    # 意思就是temperature越大不太可能的token就越容易被选中
    probs = probs ** (1.0 / temperature)
# sample from probs, 获取下标
out = torch.multinomial(probs, num_samples=1)[0]
token = int(out)

# NOTE: 重点是模型怎么处理chunk

