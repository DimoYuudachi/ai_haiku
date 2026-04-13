import os, json, random
from typing import List, Tuple, Optional, Dict
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sudachipy import tokenizer, dictionary
from sklearn.model_selection import train_test_split

# 設定（ハイパーパラメータ）
EMBEDDING_DIM = 64
HIDDEN_SIZE = 64
BATCH_SIZE = 32
EPOCHS = 20
LR = 1e-3

AUTO_MAX_LEN = True
MAX_LEN_CAP = 60
MIN_MAX_LEN = 20
PERCENTILE_FOR_MAX_LEN = 98
EXTRA_LEN_MARGIN = 2

SEED = 42
THRESHOLD = 0.77

# train/val/test の分割比（全体に対する比率）
TEST_SIZE = 0.20
VAL_SIZE = 0.10

PUNCT_SURF = {
    "、", "。", "，", "．", ",", ".", "！", "!", "？", "?", "・", "：", ":", "；", ";",
    "「", "」", "『", "』", "（", "）", "(", ")", "［", "］", "[", "]", "【", "】",
    "…", "‥", "—", "ー", "〜", "~", "‧"
}

BASE_POS = [
    "名詞", "助詞", "動詞", "形容詞", "助動詞", "副詞", "記号", "連体詞", "接続詞",
    "感動詞", "接頭詞", "接尾詞", "代名詞", "助数詞", "連語", "未定義語"
]

# 早期終了
EARLY_PATIENCE = 3
EARLY_MIN_DELTA = 1e-4

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

# 語彙
class Vocab:
    def __init__(self):
        self.token2id: Dict[str, int] = {}
        self.id2token: Dict[int, str] = {}
        self.max_len: Optional[int] = None

        self.BOS, self.EOS, self.PAD, self.UNK = "<BOS>", "<EOS>", "<PAD>", "未定義語"

        for t in (self.BOS, self.EOS, self.PAD):
            self.add(t)

        for p in BASE_POS:
            self.add(p)

        self.bos_id = self.token2id[self.BOS]
        self.eos_id = self.token2id[self.EOS]
        self.pad_id = self.token2id[self.PAD]
        self.unk_id = self.token2id[self.UNK]

    def add(self, token: str) -> int:
        if token not in self.token2id:
            idx = len(self.token2id)
            self.token2id[token] = idx
            self.id2token[idx] = token
        return self.token2id[token]

    def get_id(self, token: str) -> int:
        return self.token2id.get(token, self.unk_id)

    def __len__(self):
        return len(self.token2id)

    def save(self, path: str):
        data = {
            "token2id": self.token2id,
            "specials": {"BOS": self.BOS, "EOS": self.EOS, "PAD": self.PAD, "UNK": self.UNK},
            "max_len": self.max_len
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    @staticmethod
    def load(path: str) -> "Vocab":
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        v = Vocab()

        v.token2id = {k: int(vv) for k, vv in data["token2id"].items()}
        v.id2token = {int(vv): k for k, vv in v.token2id.items()}

        v.BOS = data["specials"]["BOS"]
        v.EOS = data["specials"]["EOS"]
        v.PAD = data["specials"]["PAD"]
        v.UNK = data["specials"]["UNK"]
        v.max_len = data.get("max_len", None)

        v.bos_id = v.token2id[v.BOS]
        v.eos_id = v.token2id[v.EOS]
        v.pad_id = v.token2id[v.PAD]
        v.unk_id = v.token2id.get(v.UNK, v.token2id.get("未定義語", 0))
        return v

# データ構築
class Builder:
    def __init__(self, csv_path: str, text_col: str = "俳句", test_size: float = TEST_SIZE, val_size: float = VAL_SIZE, seed: int = SEED):
        df = pd.read_csv(csv_path)
        if text_col not in df.columns:
            raise ValueError(f"CSVに列がありません：{text_col}")
        self.haikus = df[text_col].dropna().astype(str).tolist()

        self.test_size = float(test_size)
        self.val_size = float(val_size)
        self.seed = int(seed)

        self.sudachi = dictionary.Dictionary().create()
        self.mode = tokenizer.Tokenizer.SplitMode.C
        self.vocab = Vocab()
        self.max_len: Optional[int] = None

    # 俳句からタグ列に変換
    def tags(self, text: str, build_vocab: bool) -> List[str]:
        try:
            toks = self.sudachi.tokenize(text, self.mode)
            out: List[str] = []
            for t in toks:
                surf = t.surface()

                # 句読点は「記号」にまとめる
                if surf in PUNCT_SURF:
                    out.append("記号")
                    if build_vocab:
                        self.vocab.add("記号")
                    continue

                pos = t.part_of_speech()
                pos0 = pos[0] if pos else "未定義語"

                # 助詞は表層を残す
                if pos0 == "助詞":
                    tag = f"助詞_{surf}"
                    out.append(tag)
                    if build_vocab:
                        self.vocab.add(tag)
                    continue

                # 記号判定
                if "記号" in pos:
                    tag = "記号"
                elif pos0 in BASE_POS:
                    tag = pos0
                else:
                    tag = "未定義語"

                out.append(tag)
                if build_vocab:
                    self.vocab.add(tag)

            return out
        except Exception:
            return []

    # 最大長の推定（BOS/EOS を含める）
    def _estimate_max_len(self, train_texts: List[str]) -> int:
        lengths = []
        for s in train_texts:
            tg = self.tags(s, build_vocab=False)
            if tg:
                lengths.append(len(tg) + 2)
        if not lengths:
            return MIN_MAX_LEN
        p = int(np.percentile(lengths, PERCENTILE_FOR_MAX_LEN))
        est = max(p + EXTRA_LEN_MARGIN, MIN_MAX_LEN)
        return min(est, MAX_LEN_CAP)

    # 負例生成（タグ列を壊す）
    @staticmethod
    def _swap(tags: List[str]) -> List[str]:
        out = tags.copy()
        i, j = random.sample(range(len(out)), 2)
        out[i], out[j] = out[j], out[i]
        return out

    @staticmethod
    def _shuffle_span(tags: List[str]) -> List[str]:
        out = tags.copy()
        n = len(out)
        if n < 4:
            return Builder._swap(tags)
        a = random.randint(0, n - 3)
        b = random.randint(a + 2, min(n, a + 6))
        span = out[a:b]
        random.shuffle(span)
        out[a:b] = span
        return out

    @staticmethod
    def _drop(tags: List[str]) -> Optional[List[str]]:
        if len(tags) <= 2:
            return None
        out = tags.copy()
        out.pop(random.randrange(len(out)))
        return out

    @staticmethod
    def _dup(tags: List[str]) -> List[str]:
        out = tags.copy()
        k = random.randrange(len(out))
        out.insert(k, out[k])
        return out

    @staticmethod
    def _move_particle_block(tags: List[str]) -> Optional[List[str]]:
        out = tags.copy()
        n = len(out)
        idxs = [i for i, t in enumerate(out) if t.startswith("助詞_") or t == "助動詞"]
        if not idxs:
            return None
        i = random.choice(idxs)
        l, r = i, i + 1
        while l - 1 >= 0 and (out[l - 1].startswith("助詞_") or out[l - 1] == "助動詞"):
            l -= 1
        while r < n and (out[r].startswith("助詞_") or out[r] == "助動詞"):
            r += 1
        block = out[l:r]
        rest = out[:l] + out[r:]
        if not rest:
            return None
        return (block + rest) if random.random() < 0.5 else (rest + block)

    def corrupt(self, tags: List[str]) -> Optional[List[str]]:
        if len(tags) < 2:
            return None
        ops = [self._swap, self._shuffle_span, self._dup]
        ops_maybe = [self._drop, self._move_particle_block]
        for _ in range(6):
            fn = random.choice(ops + ops_maybe)
            out = fn(tags)
            if out is None:
                continue
            if out != tags and len(out) >= 1:
                return out
        return None

    # タグ列からID列（BOS/EOS + PAD）に変換
    def encode(self, tags: List[str], max_len: int) -> Tuple[List[int], int]:
        ids = [self.vocab.bos_id] + [self.vocab.get_id(t) for t in tags] + [self.vocab.eos_id]
        if len(ids) > max_len:
            ids = ids[:max_len]
            length = max_len
        else:
            length = len(ids)
            ids += [self.vocab.pad_id] * (max_len - len(ids))
        return ids, length

    # train/val/test を作る（語彙は train のみで構築）
    def build(self):
        # 1) test を先に切り出す
        trainval_texts, test_texts = train_test_split(
            self.haikus, test_size=self.test_size, random_state=self.seed, shuffle=True
        )

        # 2) trainval から val を切り出す
        val_ratio = self.val_size / max(1e-9, (1.0 - self.test_size))
        val_ratio = min(max(val_ratio, 0.01), 0.5)

        train_texts, val_texts = train_test_split(
            trainval_texts, test_size=val_ratio, random_state=self.seed, shuffle=True
        )

        # 3) 語彙は train のみで構築
        for s in train_texts:
            _ = self.tags(s, build_vocab=True)

        self.max_len = self._estimate_max_len(train_texts) if AUTO_MAX_LEN else MIN_MAX_LEN

        def make_samples(texts: List[str]) -> List[Tuple[List[str], int]]:
            positives = []
            for s in texts:
                tg = self.tags(s, build_vocab=False)
                if tg:
                    positives.append(tg)

            samples = [(tg, 1) for tg in positives]
            for tg in positives:
                neg = self.corrupt(tg)
                if neg:
                    samples.append((neg, 0))

            random.shuffle(samples)
            return samples

        train_samples = make_samples(train_texts)
        val_samples = make_samples(val_texts)
        test_samples = make_samples(test_texts)

        return train_samples, val_samples, test_samples, self.max_len, self.vocab

# Dataset / DataLoader
class TagSeqDataset(Dataset):
    def __init__(self, samples: List[Tuple[List[str], int]], builder: Builder, max_len: int):
        self.samples = samples
        self.builder = builder
        self.max_len = max_len

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        tags, label = self.samples[idx]
        ids, length = self.builder.encode(tags, self.max_len)
        return (
            torch.tensor(ids, dtype=torch.long),
            torch.tensor(length, dtype=torch.long),
            torch.tensor(label, dtype=torch.float32),
        )

def collate_fn(batch):
    xs, lens, ys = zip(*batch)
    return torch.stack(xs, 0), torch.stack(lens, 0), torch.stack(ys, 0)

# 評価器モデル
class HaikuEvaluatorModel(nn.Module):
    def __init__(self, vocab_size: int, pad_id: int):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, EMBEDDING_DIM, padding_idx=pad_id)
        self.lstm = nn.LSTM(EMBEDDING_DIM, HIDDEN_SIZE, batch_first=True)
        self.fc1 = nn.Linear(HIDDEN_SIZE, 32)
        self.fc2 = nn.Linear(32, 1)

    def forward(self, x, lengths):
        emb = self.embedding(x)
        packed = nn.utils.rnn.pack_padded_sequence(
            emb, lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        _, (h, _) = self.lstm(packed)
        h_last = h[-1]
        logits = self.fc2(torch.relu(self.fc1(h_last))).squeeze(-1)
        return logits

# 学習
class Trainer:
    def __init__(self, model, train_loader, val_loader, device=None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.crit = nn.BCEWithLogitsLoss()
        self.opt = optim.Adam(self.model.parameters(), lr=LR)

        self.best_state = None
        self.best_val_loss = float("inf")

    @staticmethod
    def acc_from_logits(logits, y, thr=THRESHOLD):
        probs = torch.sigmoid(logits)
        preds = (probs >= thr).float()
        return (preds == y).float().mean().item()

    def train_epoch(self):
        self.model.train()
        tot_loss, tot_acc, n = 0.0, 0.0, 0
        for x, lengths, y in self.train_loader:
            x, lengths, y = x.to(self.device), lengths.to(self.device), y.to(self.device)
            logits = self.model(x, lengths)
            loss = self.crit(logits, y)

            self.opt.zero_grad()
            loss.backward()
            self.opt.step()

            bs = x.size(0)
            tot_loss += loss.item() * bs
            tot_acc += self.acc_from_logits(logits.detach(), y) * bs
            n += bs
        return tot_loss / n, tot_acc / n

    def evaluate(self, loader):
        self.model.eval()
        tot_loss, tot_acc, n = 0.0, 0.0, 0
        with torch.no_grad():
            for x, lengths, y in loader:
                x, lengths, y = x.to(self.device), lengths.to(self.device), y.to(self.device)
                logits = self.model(x, lengths)
                loss = self.crit(logits, y)

                bs = x.size(0)
                tot_loss += loss.item() * bs
                tot_acc += self.acc_from_logits(logits, y) * bs
                n += bs
        return tot_loss / n, tot_acc / n

    def fit(self, epochs=EPOCHS, patience: int = EARLY_PATIENCE, min_delta: float = EARLY_MIN_DELTA):
        print(f"学習を開始します。使用デバイス: {self.device}")
        bad = 0

        for ep in range(1, epochs + 1):
            tr_loss, tr_acc = self.train_epoch()
            va_loss, va_acc = self.evaluate(self.val_loader)

            improved = (self.best_val_loss - va_loss) > min_delta
            if improved:
                self.best_val_loss = va_loss
                self.best_state = {k: v.detach().cpu().clone() for k, v in self.model.state_dict().items()}
                bad = 0
                mark = " *best"
            else:
                bad += 1
                mark = f" (patience {bad}/{patience})"

            print(
                f"Epoch [{ep}/{epochs}] 学習 Loss:{tr_loss:.4f} Acc:{tr_acc:.4f} / "
                f"検証 Loss:{va_loss:.4f} Acc:{va_acc:.4f}{mark}"
            )

            if bad >= patience:
                print(f"早期終了: 検証 loss が {patience} 回改善しませんでした。best_val_loss={self.best_val_loss:.4f}")
                break

        # 学習終了後に best を復元
        if self.best_state is not None:
            self.model.load_state_dict(self.best_state)

        return self.best_val_loss

# 推論用 Evaluator
class HaikuEvaluator:
    def __init__(self, model_path: str, vocab_path: str, max_len: int = None, device=None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.vocab = Vocab.load(vocab_path)
        self.max_len = max_len if max_len is not None else (self.vocab.max_len or MIN_MAX_LEN)

        self.sudachi = dictionary.Dictionary().create()
        self.mode = tokenizer.Tokenizer.SplitMode.C

        self.model = HaikuEvaluatorModel(vocab_size=len(self.vocab), pad_id=self.vocab.pad_id)
        state = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(state)
        self.model.to(self.device).eval()

    def tags(self, text: str) -> List[str]:
        try:
            toks = self.sudachi.tokenize(text, self.mode)
            out = []
            for t in toks:
                surf = t.surface()
                if surf in PUNCT_SURF:
                    out.append("記号")
                    continue
                pos = t.part_of_speech()
                pos0 = pos[0] if pos else "未定義語"
                if pos0 == "助詞":
                    out.append(f"助詞_{surf}")
                    continue
                if "記号" in pos:
                    out.append("記号")
                elif pos0 in BASE_POS:
                    out.append(pos0)
                else:
                    out.append("未定義語")
            return out
        except Exception:
            return []

    def encode(self, tags: List[str]) -> Tuple[List[int], int]:
        ids = [self.vocab.bos_id] + [self.vocab.get_id(t) for t in tags] + [self.vocab.eos_id]
        if len(ids) > self.max_len:
            ids = ids[:self.max_len]
            length = self.max_len
        else:
            length = len(ids)
            ids += [self.vocab.pad_id] * (self.max_len - len(ids))
        return ids, length

    def analyze(self, haiku: str) -> Tuple[float, List[str], str]:
        tags = self.tags(haiku)
        if not tags:
            return 0.0, [], "品詞解析に失敗しました"

        ids, length = self.encode(tags)
        x = torch.tensor([ids], dtype=torch.long, device=self.device)
        lengths = torch.tensor([length], dtype=torch.long, device=self.device)

        with torch.no_grad():
            logits = self.model(x, lengths)
            prob = torch.sigmoid(logits).item()

        result = "正常" if prob >= THRESHOLD else "異常"
        return prob, tags, result

    def batch_prob(self, haikus: List[str], batch_size: int = 256) -> List[float]:
        if not haikus:
            return []

        self.model.eval()
        out_probs: List[float] = []

        for i in range(0, len(haikus), batch_size):
            batch = haikus[i:i + batch_size]
            ids_list, len_list = [], []

            for h in batch:
                tg = self.tags(h)
                if not tg:
                    ids = [self.vocab.bos_id, self.vocab.eos_id] + [self.vocab.pad_id] * (self.max_len - 2)
                    length = 2
                else:
                    ids, length = self.encode(tg)
                ids_list.append(ids)
                len_list.append(length)

            x = torch.tensor(ids_list, dtype=torch.long, device=self.device)
            lengths = torch.tensor(len_list, dtype=torch.long, device=self.device)

            with torch.no_grad():
                logits = self.model(x, lengths)
                probs = torch.sigmoid(logits).detach().cpu().numpy().tolist()

            out_probs.extend(probs)

        return out_probs

    def batch_is_normal(self, haikus: List[str], batch_size: int = 256, threshold: float = THRESHOLD) -> List[bool]:
        probs = self.batch_prob(haikus, batch_size=batch_size)
        return [p >= threshold for p in probs]

# 実行
def main():
    csv_path = "data/shiki_with_seasons.csv"
    model_path = "model/haiku_evaluator_model.pth"
    vocab_path = "model/haiku_vocab.json"
    os.makedirs("model", exist_ok=True)

    builder = Builder(csv_path, test_size=TEST_SIZE, val_size=VAL_SIZE, seed=SEED)
    train_samples, val_samples, test_samples, max_len, vocab = builder.build()

    pos_tr = sum(1 for _, y in train_samples if y == 1)
    neg_tr = sum(1 for _, y in train_samples if y == 0)
    pos_va = sum(1 for _, y in val_samples if y == 1)
    neg_va = sum(1 for _, y in val_samples if y == 0)
    pos_te = sum(1 for _, y in test_samples if y == 1)
    neg_te = sum(1 for _, y in test_samples if y == 0)

    print("データを前処理しています...")
    print(f"学習：正例 {pos_tr} / 負例 {neg_tr}")
    print(f"検証：正例 {pos_va} / 負例 {neg_va}")
    print(f"評価：正例 {pos_te} / 負例 {neg_te}")
    print(f"max_len（BOS/EOS含む）= {max_len}")
    print(f"vocab_size = {len(vocab)}（助詞は表層も含む）")

    vocab.max_len = max_len
    vocab.save(vocab_path)
    print(f"語彙を保存しました: {vocab_path}")

    train_ds = TagSeqDataset(train_samples, builder, max_len)
    val_ds = TagSeqDataset(val_samples, builder, max_len)
    test_ds = TagSeqDataset(test_samples, builder, max_len)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)

    model = HaikuEvaluatorModel(vocab_size=len(vocab), pad_id=vocab.pad_id)
    trainer = Trainer(model, train_loader, val_loader)

    trainer.fit(epochs=EPOCHS, patience=EARLY_PATIENCE, min_delta=EARLY_MIN_DELTA)

    # best（検証最小 loss）を保存
    torch.save(trainer.model.state_dict(), model_path)
    print(f"モデルを保存しました（best 検証 loss）: {model_path}")

    # test は学習後に 1 回だけ評価
    test_loss, test_acc = trainer.evaluate(test_loader)
    print(f"\n最終 評価結果（test）: Loss:{test_loss:.4f} Acc:{test_acc:.4f}（threshold={THRESHOLD}）")

    evaluator = HaikuEvaluator(model_path, vocab_path, max_len=None)
    test_haikus = [
        "春が来た、夏が来る、秋が去る",
        "来た春が、来る夏が、去る秋が",
        "秋高し雲より上を鳥かける",
        "雲より上を鳥かける秋高し"
    ]

    print(f"\n=== 評価器の使用例（threshold={THRESHOLD}）===")
    for h in test_haikus:
        prob, tags, result = evaluator.analyze(h)
        print(f"俳句: {h}")
        print(f"tag列: {tags}")
        print(f"正常確率: {prob:.4f} → 判定: {result}")
        print("-" * 50)

if __name__ == "__main__":
    main()

