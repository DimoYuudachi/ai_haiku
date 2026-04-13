import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sudachipy import tokenizer
from sudachipy import dictionary
import os
from collections import Counter
import random
import json
import requests
import gzip
import pyopenjtalk

# ======================
# 固定乱数シード
# ======================
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

# ======================
# fastText 事前学習ベクトル
# ======================
def download_pretrained_vectors():
    if not os.path.exists('ja.vec'):
        print("fastText（日本語）ベクトルをダウンロードしています…")
        url = "https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.ja.300.vec.gz"
        response = requests.get(url, stream=True)
        
        with open('cc.ja.300.vec.gz', 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        
        with gzip.open('cc.ja.300.vec.gz', 'rb') as f_in:
            with open('ja.vec', 'wb') as f_out:
                f_out.write(f_in.read())
        
        os.remove('cc.ja.300.vec.gz')
        print("fastText の準備が完了しました。")
    return 'ja.vec'

def load_pretrained_vectors(vec_file, word_to_idx, embed_dim=300):
    """ 語彙に合う行だけ読み込み、埋め込み行列を作る """
    embedding = np.random.uniform(-0.25, 0.25, (len(word_to_idx), embed_dim))
    found = 0

    with open(vec_file, 'r', encoding='utf-8', errors='ignore') as f:
        f.readline()
        for line in f:
            parts = line.strip().split()
            if len(parts) != embed_dim + 1:
                continue
            word = parts[0]
            if word in word_to_idx:
                idx = word_to_idx[word]
                embedding[idx] = np.array(parts[1:], dtype=np.float32)
                found += 1

    print(f"fastText ベクトル一致: {found} / {len(word_to_idx)}")
    return torch.tensor(embedding, dtype=torch.float32)

# ======================
# 設定保存
# ======================
def save_haiku_config(
    config_path: str,
    word_to_idx: dict,
    idx_to_word: dict,
    *,
    embed_dim: int,
    hidden_dim: int,
    num_layers: int,
    input_dropout: float,
    layer_dropout: float,
    max_length: int,
    min_freq: int,
):
    idx_to_word_json = {str(int(k)): v for k, v in idx_to_word.items()}

    cfg = {
        "vocab_size": len(word_to_idx),
        "word_to_idx": word_to_idx,
        "idx_to_word": idx_to_word_json,
        "embed_dim": embed_dim,
        "hidden_dim": hidden_dim,
        "num_layers": num_layers,
        "input_dropout": input_dropout,
        "layer_dropout": layer_dropout,
        "max_length": max_length,
        "min_freq": min_freq,
        "special_tokens": {
            "PAD": "<PAD>",
            "UNK": "<UNK>",
            "START": "<START>",
            "END": "<END>",
            "SEP": "<SEP>",
        }
    }

    os.makedirs(os.path.dirname(config_path), exist_ok=True)
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(cfg, f, ensure_ascii=False, indent=2)

    print(f"設定ファイルを保存しました: {config_path} (vocab={len(word_to_idx)})")

# ======================
# モーラカウンタ
# ======================
class MoraCounter:
    def count_mora(self, text):
        if not text:
            return 0

        kana = pyopenjtalk.g2p(text, kana=True).replace(" ", "")
        kana = kana.replace(" ", "")
        small = set("ァィゥェォャュョぁぃぅぇぉゃゅょゎヮ")
        count = 0
        for c in kana:
            if c in small:
                continue
            if ('ァ' <= c <= 'ヴ') or ('ぁ' <= c <= 'ゖ') or c == 'ー':
                count += 1
        return count

# ======================
# 分かち書き
# ======================
def tokenize_japanese(text, tok):
    mode = tokenizer.Tokenizer.SplitMode.B
    return [t.surface() for t in tok.tokenize(text, mode)]

# ======================
# 語彙作成
# ======================
def build_vocab(tokenized, min_freq=3):
    counter = Counter()
    for h in tokenized:
        counter.update(h)

    words = [w for w,c in counter.items() if c >= min_freq]

    vocab = ['<PAD>', '<UNK>', '<START>', '<END>', '<SEP>'] + sorted(words)

    w2i = {w:i for i,w in enumerate(vocab)}
    i2w = {i: w for i, w in enumerate(vocab)}

    print(f"総単語数={len(counter)}，語彙サイズ={len(vocab)}")
    return w2i, i2w


# ======================
# 5-7-5 構造化データセット
# ======================
class StructuredHaikuDataset(Dataset):
    def __init__(self, haikus, w2i, mora, max_length=30):
        self.w2i = w2i
        self.mora = mora
        self.max_length = max_length
        self.samples = []

        for h in haikus:
            lines = self._split_575(h)
            if lines:
                self.samples.append(lines)

        print(f"5-7-5 として使えるサンプル数：{len(self.samples)}")

    def _split_575(self, tokens):
        text = ''.join(tokens)
        m = self.mora.count_mora(text)
        if m < 15 or m > 21:
            return None
        
        pos = 0
        result = []
        targets = [5,7,5]

        for t in targets:
            cur = []
            line_text = ""
            last_mora = 0
            while pos < len(tokens) and last_mora < t:
                w = tokens[pos]
                new_text = line_text + w
                new_mora = self.mora.count_mora(new_text)
                delta = new_mora - last_mora
                if delta <= 0:
                    pos += 1
                    continue
                if new_mora > t:
                    pos += 1
                    continue
                
                cur.append(w)
                line_text = new_text
                last_mora = new_mora
                pos += 1
            
            if last_mora != t:
                return None
            
            result.append(cur)

        return result if len(result)==3 else None

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        lines = self.samples[idx]
        seq = ['<START>']
        for i, line in enumerate(lines):
            seq.extend(line)
            if i < 2:
                seq.append('<SEP>')
        seq.append('<END>')

        ids = [self.w2i.get(w, self.w2i['<UNK>']) for w in seq]

        if len(ids) < self.max_length:
            pad = self.max_length - len(ids)
            inp = ids[:-1] + [self.w2i['<PAD>']]*pad
            tgt = ids[1:] + [self.w2i['<PAD>']]*pad
        else:
            inp = ids[:self.max_length]
            tgt = ids[1:self.max_length+1]

        return torch.tensor(inp), torch.tensor(tgt)

# ======================
# モデル
# ======================
class BigLSTMGenerator(nn.Module):
    def __init__(self, vocab, embed_dim=200, hidden_dim=256, layers=2, input_dropout=0.1, layer_dropout=0.1):
        super().__init__()
        self.embedding = nn.Embedding(vocab, embed_dim, padding_idx=0)
        self.in_drop = nn.Dropout(input_dropout)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, layers,
                            batch_first=True, dropout=layer_dropout if layers > 1 else 0.0)
        self.fc = nn.Linear(hidden_dim, vocab)

    def forward(self, x, hidden=None):
        e = self.embedding(x)
        e = self.in_drop(e)
        o, hidden = self.lstm(e, hidden)
        o = self.fc(o)
        return o, hidden


# ======================
# 学習
# ======================
def train_model(model, train_loader, val_loader, device, epochs=1000, save_path="model/haiku_generator_best.pt", patience=10, min_delta=0.0):
    model.to(device)
    crit = nn.CrossEntropyLoss(ignore_index=0)
    opt = optim.Adam(model.parameters(), lr=0.001)

    best_val = float('inf')
    bad_epochs = 0

    for ep in range(epochs):
        model.train()
        total = 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)

            opt.zero_grad()
            out, _ = model(x)
            loss = crit(out.reshape(-1, model.fc.out_features), y.reshape(-1))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

            total += loss.item()

        train_loss = total / len(train_loader)

        model.eval()
        val_total = 0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                out, _ = model(x)
                loss = crit(out.reshape(-1, model.fc.out_features), y.reshape(-1))
                val_total += loss.item()

        val_loss = val_total / len(val_loader)

        print(f"[エポック {ep+1}] train={train_loss:.4f}  val={val_loss:.4f}")

        if val_loss < best_val - min_delta:
            best_val = val_loss
            bad_epochs = 0
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            torch.save(model.state_dict(), save_path)
            print(f" ベスト更新(val={best_val:.4f})")
        else:
            bad_epochs += 1
            if bad_epochs >= patience:
                print(f" 早期終了：検証損失が {patience} エポック改善しませんでした（best={best_val:.4f}）")
                break

    print("学習終了。最良モデル:", save_path)
    return model

# ======================
# 俳句生成
# ======================
def generate_haiku(model, start_word, w2i, i2w, mora, device, temperature=0.8):
    model.eval()
    model.to(device)

    if start_word not in w2i:
        start_word = '<START>'
    
    generated_tokens = [w2i['<START>']]
    if start_word != '<START>':
        generated_tokens.append(w2i[start_word])
    
    hidden = None
    lines = [[], [], []]
    targets = [5, 7, 5]
    
    for L in range(3):
        current_mora = 0
        if L == 0 and start_word != '<START>':
            lines[0].append(start_word)
            current_mora = mora.count_mora(start_word)
        
        max_attempts = 100
        attempts = 0
        
        while current_mora < targets[L] and attempts < max_attempts:
            attempts += 1
            
            x = torch.tensor([[generated_tokens[-1]]]).to(device)
            
            with torch.no_grad():
                out, hidden = model(x, hidden)
                logits = out[0, -1] / temperature
                probs = torch.softmax(logits, dim=-1)
            
            special_tokens = {w2i['<PAD>'], w2i['<UNK>'], w2i['<START>'], w2i['<END>'], w2i['<SEP>']}
            
            for _ in range(20):
                next_idx = torch.multinomial(probs, 1).item()
                
                if next_idx in special_tokens:
                    continue
                
                word = i2w[next_idx]
                word_mora = mora.count_mora(word)
                
                if current_mora + word_mora <= targets[L]:
                    lines[L].append(word)
                    generated_tokens.append(next_idx)
                    current_mora += word_mora
                    break
            else:
                top_k = torch.topk(probs, k=50)
                for idx in top_k.indices:
                    idx = idx.item()
                    if idx in special_tokens:
                        continue
                    word = i2w[idx]
                    word_mora = mora.count_mora(word)
                    if word_mora > 0 and current_mora + word_mora <= targets[L]:
                        lines[L].append(word)
                        generated_tokens.append(idx)
                        current_mora += word_mora
                        break
        
        if current_mora == targets[L] - 1:
            fillers = ['や', 'かな', 'けり', 'ぞ', 'や']
            for filler in fillers:
                if filler in w2i:
                    lines[L].append(filler)
                    generated_tokens.append(w2i[filler])
                    current_mora += mora.count_mora(filler)
                    break
        
        if L < 2:
            sep_idx = w2i['<SEP>']
            generated_tokens.append(sep_idx)
            x = torch.tensor([[sep_idx]]).to(device)
            with torch.no_grad():
                _, hidden = model(x, hidden)
    
    result_lines = []
    for line_tokens in lines:
        if line_tokens:
            result_lines.append(''.join(line_tokens))
        else:
            result_lines.append('...')
    
    return '\n'.join(result_lines)

# ======================
# メイン
# ======================
def main():
    print("Sudachi 分かち書き器を初期化します…")
    tok = dictionary.Dictionary().create()
    mora = MoraCounter()

    print("データを読み込みます…")
    df = pd.read_csv('data/shiki_merged.csv')
    texts = df['俳句'].tolist()

    tokenized = []
    for t in texts:
        try:
            tokenized.append(tokenize_japanese(t, tok))
        except:
            pass

    print("語彙を作成します…")
    w2i, i2w = build_vocab(tokenized, min_freq=3)

    print("fastText を準備します…")
    vec = download_pretrained_vectors()

    print("fastText ベクトルを読み込みます（300次元）…")
    emb_300 = load_pretrained_vectors(vec, w2i, embed_dim=300)

    print("300次元 → 200次元へ射影します…")
    torch.manual_seed(SEED)
    projection = torch.randn(300, 200) * 0.01
    torch.save(projection, "model/fasttext_proj_300to200.pt")
    emb_200 = torch.mm(emb_300, projection)

    print("5-7-5 データセットを構築します…")
    full = StructuredHaikuDataset(tokenized, w2i, mora, max_length=30)

    CONFIG_PATH = "data/haiku_config_v2.json"

    EMBED_DIM = 200
    HIDDEN_DIM = 256
    NUM_LAYERS = 2
    INPUT_DROPOUT = 0.1
    LAYER_DROPOUT = 0.1
    MAX_LENGTH = 30
    MIN_FREQ = 3

    save_haiku_config(
    CONFIG_PATH,
    word_to_idx=w2i,
    idx_to_word=i2w,
    embed_dim=EMBED_DIM,
    hidden_dim=HIDDEN_DIM,
    num_layers=NUM_LAYERS,
    input_dropout=INPUT_DROPOUT,
    layer_dropout=LAYER_DROPOUT,
    max_length=MAX_LENGTH,
    min_freq=MIN_FREQ,
)

    train_size = int(len(full)*0.85)
    val_size = len(full) - train_size

    train, val = torch.utils.data.random_split(full,[train_size,val_size])

    train_loader = DataLoader(train,batch_size=32,shuffle=True)
    val_loader = DataLoader(val,batch_size=32)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"デバイス: {device}")

    print("モデルを作成します…")
    model = BigLSTMGenerator(len(w2i), embed_dim=200, hidden_dim=256, layers=2, input_dropout=0.1, layer_dropout=0.1)
    model.embedding.weight = nn.Parameter(emb_200)

    print("学習を開始します…")
    train_model(
    model,
    train_loader,
    val_loader,
    device,
    epochs=1000,
    save_path="model/haiku_generator_best.pt",
    patience=10,
    min_delta=0.0
)
    print("\n生成例（5-7-5）：")
    for w in ("秋", "月", "桜"):
        print(f"\n開始語: {w}")
        print(generate_haiku(model, w, w2i, i2w, mora, device, temperature=0.8))
        
if __name__=="__main__":
    main()