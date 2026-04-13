import json
from functools import lru_cache
from typing import List, Tuple

import torch
import torch.nn as nn
from sudachipy import tokenizer, dictionary

# ============================================================
# 定数
# ============================================================
EMBEDDING_DIM = 64
HIDDEN_SIZE = 64

PUNCT_SURF = {
    "、", "。", "，", "．", ",", ".", "！", "!", "？", "?", "・", "：", ":", "；", ";",
    "「", "」", "『", "』", "（", "）", "(", ")", "［", "］", "[", "]", "【", "】",
    "…", "‥", "—", "ー", "〜", "~", "‧"
}

BASE_POS = [
    "名詞", "助詞", "動詞", "形容詞", "助動詞", "副詞", "記号", "連体詞", "接続詞",
    "感動詞", "接頭詞", "接尾詞", "代名詞", "助数詞", "連語", "未定義語"
]

# ============================================================
# Sudachi
# ============================================================

@lru_cache(maxsize=1)
def get_sudachi():
    tok = dictionary.Dictionary().create()
    mode = tokenizer.Tokenizer.SplitMode.C
    return tok, mode

# ============================================================
# 語彙
# ============================================================
class Vocab:
    def __init__(self, token2id, specials):
        self.token2id = {k: int(v) for k, v in token2id.items()}
        self.id2token = {int(v): k for k, v in self.token2id.items()}
        self.BOS = specials["BOS"]
        self.EOS = specials["EOS"]
        self.PAD = specials["PAD"]
        self.UNK = specials["UNK"]
        self.bos_id = self.token2id[self.BOS]
        self.eos_id = self.token2id[self.EOS]
        self.pad_id = self.token2id[self.PAD]
        self.unk_id = self.token2id[self.UNK]

    @staticmethod
    def load(path: str) -> "Vocab":
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return Vocab(data["token2id"], data["specials"])

    def get_id(self, token: str) -> int:
        return self.token2id.get(token, self.unk_id)

    def __len__(self):
        return len(self.token2id)

# ============================================================
# 評価モデル
# ============================================================
class HaikuEvaluatorModel(nn.Module):
    def __init__(self, vocab_size: int, pad_id: int, embedding_dim=EMBEDDING_DIM, hidden_size=HIDDEN_SIZE):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_id)
        self.lstm = nn.LSTM(embedding_dim, hidden_size, batch_first=True)
        self.fc1 = nn.Linear(hidden_size, 32)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(32, 1)

    def forward(self, x, lengths):
        emb = self.embedding(x)
        packed = nn.utils.rnn.pack_padded_sequence(
            emb, lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        _, (h_n, _) = self.lstm(packed)
        last_hidden = h_n[-1]  # (B, H)
        logits = self.fc2(self.relu(self.fc1(last_hidden))).squeeze(-1)  # (B,)
        return logits

# ============================================================
# 評価器
# ============================================================
class HaikuEvaluator:
    def __init__(self, model_path: str, vocab_path: str, max_len: int, threshold: float = 0.5, device=None):
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.threshold = float(threshold)
        self.max_len = int(max_len)

        self.vocab = Vocab.load(vocab_path)

        self.sudachi = dictionary.Dictionary().create()
        self.mode = tokenizer.Tokenizer.SplitMode.C

        self.model = HaikuEvaluatorModel(vocab_size=len(self.vocab), pad_id=self.vocab.pad_id)
        state = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(state)
        self.model.to(self.device).eval()

    def _analyze_tags(self, text: str) -> List[str]:
        tokens = self.sudachi.tokenize(text, self.mode)
        tags = []
        for t in tokens:
            surf = t.surface()
            if surf in PUNCT_SURF:
                tags.append("記号")
                continue

            pos = t.part_of_speech()
            pos0 = pos[0] if pos else "未定義語"

            if pos0 == "助詞":
                tags.append(f"助詞_{surf}")
            elif "記号" in pos:
                tags.append("記号")
            elif pos0 in BASE_POS:
                tags.append(pos0)
            else:
                tags.append("未定義語")
        return tags

    def _encode(self, tags: List[str]) -> Tuple[torch.Tensor, torch.Tensor]:
        ids = [self.vocab.bos_id] + [self.vocab.get_id(t) for t in tags] + [self.vocab.eos_id]
        if len(ids) > self.max_len:
            ids = ids[:self.max_len]
            length = self.max_len
        else:
            length = len(ids)
            ids += [self.vocab.pad_id] * (self.max_len - len(ids))

        x = torch.tensor([ids], dtype=torch.long, device=self.device)
        lengths = torch.tensor([length], dtype=torch.long, device=self.device)
        return x, lengths

    def evaluate(self, haiku: str):
        tags = self._analyze_tags(haiku)
        if not tags:
            return 0.0, "異常", []

        x, lengths = self._encode(tags)
        with torch.no_grad():
            logits = self.model(x, lengths)
            prob = torch.sigmoid(logits).item()

        result = "正常" if prob >= self.threshold else "異常"
        return prob, result, tags


if __name__ == "__main__":
    evaluator = HaikuEvaluator(
        model_path="model/haiku_evaluator_model.pth",
        vocab_path="model/haiku_vocab.json",
        max_len=20,
        threshold=0.65
    )

    while True:
        s = input("俳句を入力してください: ").strip()
        if not s:
            continue
        if s.lower() == "exit":
            break
        prob, result, tags = evaluator.evaluate(s)
        print(f"判定: {result}  正常確率: {prob:.4f}")
        print(f"タグ列: {tags}")
        print("-" * 50)
