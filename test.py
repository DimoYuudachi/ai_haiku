import pickle
import re

with open("dataset/word2id.pkl", "rb") as f:
    word2id = pickle.load(f)

print("词表大小：", len(word2id))
print("前20个词：", list(word2id.keys())[:20])

for word in word2id:
    if re.match(r"^[0-9]+$", word):  # 匹配纯数字
        print(word)