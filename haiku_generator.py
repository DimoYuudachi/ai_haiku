import os
import glob
import json
import random
from functools import lru_cache
from typing import List, Tuple, Dict, Optional

import torch
import torch.nn as nn
from pykakasi import kakasi
from sudachipy import dictionary, tokenizer

from japanese_haiku_evaluator import HaikuEvaluator

# ============================================================
# 定数
# ============================================================
SPECIAL_TOKENS = {"<PAD>", "<UNK>", "<START>", "<END>", "<SEP>"}
KIREJI = {"や", "かな", "けり"}
INVALID_KIGO = {"雑"}
PARTICLES = {"の", "が", "を", "に", "へ", "で", "から", "まで"}

GARBAGE_PATTERNS = {
    "ゝ", "ゞ", "ゐ", "ゑ", "ヰ", "ヱ", "�",
    "哉哉", "すな", "くゝ", "かなす", "あつゝ",
}

KANJI_NUMS = set("一二三四五六七八九十百千万億兆〇零")
TARGETS_575 = (5, 7, 5)


def is_special(w: str) -> bool:
    return (not w) or (w in SPECIAL_TOKENS)


def is_text_clean(text: str) -> bool:
    if not text:
        return False
    return all(p not in text for p in GARBAGE_PATTERNS)


def count_kireji(text: str) -> int:
    return sum(text.count(k) for k in KIREJI)


def line_ends_with_particle(line: str) -> bool:
    if not line:
        return False
    for p in sorted(PARTICLES, key=len, reverse=True):
        if line.endswith(p):
            return True
    return False


# ============================================================
# Sudachi
# ============================================================
@lru_cache(maxsize=1)
def get_sudachi():
    tok = dictionary.Dictionary().create()
    mode = tokenizer.Tokenizer.SplitMode.C
    return tok, mode


def sudachi_tokens(text: str):
    tok, mode = get_sudachi()
    return tok.tokenize(text, mode)


def extract_surfaces(text: str) -> List[str]:
    return [t.surface() for t in sudachi_tokens(text)]


# ============================================================
# モーラカウンタ（pykakasi）
# ============================================================

class MoraCounter:
    def __init__(self):
        self.cache: Dict[str, int] = {}
        self.kks = kakasi()
        self.kks.setMode("J", "H")  # 漢字→ひらがな
        self.kks.setMode("K", "H")  # カタカナ→ひらがな
        self.kks.setMode("H", "H")  # ひらがな→ひらがな（維持）

        self.small_kana = set("ゃゅょぁぃぅぇぉゎャュョァィゥェォヮゕゖっッ")
        self.base_kana = set(
            "ぁあぃいぅうぇえぉお"
            "かきくけこさしすせそたちつてと"
            "なにぬねのはひふへほまみむめも"
            "やゆよらりるれろわをん"
            "がぎぐげござじずぜぞだぢづでど"
            "ばびぶべぼぱぴぷぺぽ"
            "ゃゅょゎっー"
        )

    def to_hiragana(self, text: str) -> str:
        if not text:
            return ""
        result = self.kks.convert(text)
        return "".join(item["hira"] for item in result)

    def count_mora(self, japanese_text: str) -> int:
        if not japanese_text:
            return 0
        if japanese_text in self.cache:
            return self.cache[japanese_text]

        hira = self.to_hiragana(japanese_text)
        if not hira:
            self.cache[japanese_text] = 0
            return 0

        count = 0
        i = 0
        n = len(hira)

        while i < n:
            ch = hira[i]

            # 長音は 1 モーラ
            if ch == "ー":
                count += 1
                i += 1
                continue

            # 拗音：次が小書きならまとめて 1 モーラ
            if i + 1 < n and hira[i + 1] in self.small_kana:
                count += 1
                i += 2
                continue

            # 小書きが単独で出たら無視
            if ch in self.small_kana:
                i += 1
                continue

            if ch in self.base_kana:
                count += 1

            i += 1

        self.cache[japanese_text] = count
        return count


# ============================================================
# 生成モデル
# ============================================================

class BigLSTMGenerator(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        embed_dim: int = 200,
        hidden_dim: int = 256,
        num_layers: int = 2,
        input_dropout: float = 0.1,
        layer_dropout: float = 0.1,
        pad_idx: int = 0,
    ):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_idx)
        self.in_drop = nn.Dropout(input_dropout)
        self.lstm = nn.LSTM(
            embed_dim,
            hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=layer_dropout if num_layers > 1 else 0.0,
        )
        self.fc = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x, hidden=None):
        e = self.in_drop(self.embedding(x))
        o, hidden = self.lstm(e, hidden)
        return self.fc(o), hidden


# ============================================================
# モデル・設定読み込み
# ============================================================
def find_latest_checkpoint(model_dir: str = "model", pattern: str = "*.pt") -> str:
    paths = glob.glob(os.path.join(model_dir, pattern))
    if not paths:
        raise FileNotFoundError(f"モデルが見つかりません: {model_dir}/{pattern}")
    paths.sort(key=os.path.getmtime, reverse=True)
    return paths[0]


def load_model_and_vocab(
    config_path: str = "data/haiku_config_v2.json",
    model_path: Optional[str] = None,
):
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = json.load(f)

    word_to_idx = cfg["word_to_idx"]
    idx_to_word = {int(k): v for k, v in cfg["idx_to_word"].items()}
    vocab_size = int(cfg["vocab_size"])

    model = BigLSTMGenerator(
        vocab_size=vocab_size,
        embed_dim=int(cfg.get("embed_dim", 200)),
        hidden_dim=int(cfg.get("hidden_dim", 256)),
        num_layers=int(cfg.get("num_layers", 2)),
        input_dropout=float(cfg.get("input_dropout", 0.1)),
        layer_dropout=float(cfg.get("layer_dropout", 0.1)),
        pad_idx=int(word_to_idx.get("<PAD>", 0)),
    )

    if model_path is None:
        model_path = find_latest_checkpoint("model", "*.pt")

    state = torch.load(model_path, map_location="cpu")
    model.load_state_dict(state)

    print(f"モデルを読み込みました: {model_path}")
    return model, word_to_idx, idx_to_word


# ============================================================
# 生成ユーティリティ
# ============================================================
def build_word_mora_table(word_to_idx: dict, mora: MoraCounter) -> dict:
    return {w: (0 if w in SPECIAL_TOKENS else mora.count_mora(w)) for w in word_to_idx.keys()}


def pick_start_word(
    word_to_idx: dict,
    word_mora: dict,
    target_mora: int,
    forbid_particles: bool = True,
) -> str:
    cands = [
        w for w in word_to_idx.keys()
        if (not is_special(w))
        and (not forbid_particles or w not in PARTICLES)
        and is_text_clean(w)
        and (0 < word_mora.get(w, 0) <= target_mora)
    ]
    if cands:
        return random.choice(cands)

    pool = [w for w in word_to_idx.keys() if not is_special(w)]
    return random.choice(pool) if pool else "<UNK>"


def is_valid_next_token(
    cand_word: str,
    generated: str,
    current_mora: int,
    target_mora: int,
    word_mora: dict,
    is_first_token: bool,
    forbid_particle_start: bool,
) -> bool:
    if is_special(cand_word) or (not is_text_clean(cand_word)):
        return False
    if forbid_particle_start and is_first_token and cand_word in PARTICLES:
        return False

    if generated and cand_word and generated[-1] == cand_word[0]:
        return False

    # 切れ字が 2 個以上なら拒否
    if count_kireji(generated + cand_word) > 1:
        return False

    m = word_mora.get(cand_word, 0)
    if m <= 0 or (current_mora + m > target_mora):
        return False

    # ちょうど満たすときだけ末尾助詞を禁止
    if current_mora + m == target_mora and line_ends_with_particle(generated + cand_word):
        return False

    return True


def generate_line(
    model,
    start_word: str,
    word_to_idx: dict,
    idx_to_word: dict,
    word_mora: dict,
    target_mora: int,
    device,
    *,
    temperature: float = 0.7,
    forbid_particle_start: bool = True,
    topk: int = 120,
    max_steps: int = 60,
) -> Tuple[str, int, Optional[Tuple[torch.Tensor, torch.Tensor]]]:

    if start_word not in word_to_idx:
        return "", 0, None

    idx = word_to_idx[start_word]
    x = torch.tensor([[idx]], dtype=torch.long, device=device)

    hidden = None
    generated = start_word
    current_mora = word_mora.get(start_word, 0)
    is_first_token = True

    for _ in range(max_steps):
        if current_mora >= target_mora:
            break

        with torch.inference_mode():
            out, hidden = model(x, hidden)
            logits = out[:, -1, :].squeeze(0) / max(temperature, 1e-6)

            k = min(topk, logits.numel())
            topv, topi = torch.topk(logits, k=k)
            probs = torch.softmax(topv, dim=-1)

            picked = None
            for _try in range(12):
                j = torch.multinomial(probs, 1).item()
                cand_idx = topi[j].item()
                cand_word = idx_to_word.get(cand_idx, "")

                if is_valid_next_token(
                    cand_word=cand_word,
                    generated=generated,
                    current_mora=current_mora,
                    target_mora=target_mora,
                    word_mora=word_mora,
                    is_first_token=is_first_token,
                    forbid_particle_start=forbid_particle_start,
                ):
                    picked = (cand_idx, cand_word)
                    break

            if picked is None:
                break

            next_idx, next_word = picked
            generated += next_word
            current_mora += word_mora.get(next_word, 0)
            is_first_token = False
            x = torch.tensor([[next_idx]], dtype=torch.long, device=device)

    return generated, current_mora, hidden


def feed_sep(model, word_to_idx: dict, hidden, device):
    sep_idx = word_to_idx.get("<SEP>")
    if sep_idx is None:
        return hidden
    x = torch.tensor([[sep_idx]], dtype=torch.long, device=device)
    with torch.inference_mode():
        _, hidden = model(x, hidden)
    return hidden


def generate_haiku(
    model,
    kigo: str,
    word_to_idx: dict,
    idx_to_word: dict,
    word_mora: dict,
    device,
    *,
    temperature: float = 0.7,
) -> Tuple[str, str]:
    hidden = None

    # 1句目：季語を必ず入れる
    line1, _, hidden = generate_line(
        model, kigo, word_to_idx, idx_to_word, word_mora, 5, device,
        temperature=temperature, forbid_particle_start=False
    )
    hidden = feed_sep(model, word_to_idx, hidden, device)

    start2 = pick_start_word(word_to_idx, word_mora, 7, forbid_particles=True)
    line2, _, hidden = generate_line(
        model, start2, word_to_idx, idx_to_word, word_mora, 7, device,
        temperature=temperature, forbid_particle_start=True
    )
    hidden = feed_sep(model, word_to_idx, hidden, device)

    start3 = pick_start_word(word_to_idx, word_mora, 5, forbid_particles=True)
    line3, _, _ = generate_line(
        model, start3, word_to_idx, idx_to_word, word_mora, 5, device,
        temperature=temperature, forbid_particle_start=True
    )

    formatted = f"{line1}\n{line2}\n{line3}"
    raw = line1 + line2 + line3
    return formatted, raw


def model_candidates(
    model,
    kigo: str,
    word_to_idx: dict,
    idx_to_word: dict,
    word_mora: dict,
    device,
    *,
    num_candidates: int = 300,
) -> List[Tuple[str, str]]:
    cands: List[Tuple[str, str]] = []
    temps = (0.6, 0.7, 0.8, 0.9)
    per = max(1, num_candidates // len(temps))

    for t in temps:
        for _ in range(per):
            f, r = generate_haiku(
                model, kigo, word_to_idx, idx_to_word, word_mora, device, temperature=t
            )
            if f and r:
                cands.append((f, r))
        if len(cands) >= num_candidates:
            break

    return cands[:num_candidates]


# ============================================================
# スコアリング
# ============================================================
class HaikuScorer:
    def __init__(self, mora_counter: MoraCounter):
        self.mora = mora_counter

    def _has_invalid_kireji_usage(self, formatted_text: str) -> bool:
        lines = formatted_text.split("\n")[:3]
        for line in lines:
            if not line:
                continue
            if "かな" in line and not line.endswith("かな"):
                return True
            if "けり" in line and not line.endswith("けり"):
                return True
        return False

    def _check_repetition(self, text: str) -> bool:
        run = 1
        for i in range(1, len(text)):
            if text[i] == text[i - 1]:
                run += 1
                if run >= 3:
                    return False
            else:
                run = 1
        return True

    def _check_particles_start_by_lines(self, formatted_text: str) -> bool:
        lines = formatted_text.split("\n")[:3]
        for line in lines:
            if not line:
                continue
            toks = sudachi_tokens(line)
            for t in toks:
                if t.part_of_speech()[0] == "記号":
                    continue
                if t.part_of_speech()[0] == "助詞":
                    return False
                break
        return True

    def _detect_number_penalty(self, text: str) -> float:
        if any(ch.isdigit() for ch in text):
            return 0.12
        if any(ch in KANJI_NUMS for ch in text):
            return 0.08
        return 0.0

    def _repetition_ratio(self, text: str) -> float:
        chars = [c for c in text if c.strip()]
        if not chars:
            return 1.0
        return len(set(chars)) / len(chars)

    def _noun_ratio_penalty(self, raw_text: str) -> float:
        toks = sudachi_tokens(raw_text)
        n_noun = 0
        n_all = 0
        for t in toks:
            pos0 = t.part_of_speech()[0]
            if pos0 == "記号":
                continue
            n_all += 1
            if pos0 == "名詞":
                n_noun += 1
        if n_all == 0:
            return 0.0
        ratio = n_noun / n_all
        if ratio <= 0.70:
            return 0.0
        if ratio <= 0.85:
            return (ratio - 0.70) * 0.30
        return 0.06 + (ratio - 0.85) * 0.60

    def _has_kigo(self, raw_text: str, kigo_hint: str) -> bool:
        return bool(kigo_hint) and (kigo_hint in set(extract_surfaces(raw_text)))

    def _kireji_position_bonus(self, formatted_text: str) -> float:
        lines = formatted_text.split("\n")[:3]
        if len(lines) < 3:
            return 0.0
        bonus = 0.0
        for i in (0, 1):
            for k in ("や", "かな", "けり"):
                if lines[i].endswith(k):
                    bonus += 0.05
        if lines[2].endswith("かな") or lines[2].endswith("けり"):
            bonus += 0.03
        return bonus

    @staticmethod
    def _soft_cap(x: float, cap: float, strength: float = 6.0) -> float:
        if x <= cap:
            return x
        excess = x - cap
        return cap + (excess / (1.0 + excess * strength))

    def score(self, formatted_text: str, raw_text: str, kigo_hint: Optional[str] = None) -> float:
        if (not formatted_text) or (not raw_text):
            return 0.0
        if (not is_text_clean(raw_text)) or (not is_text_clean(formatted_text)):
            return 0.0
        if self._has_invalid_kireji_usage(formatted_text):
            return 0.0

        lines = formatted_text.split("\n")[:3]
        if len(lines) < 3:
            return 0.0

        strict_575 = True
        too_far = False
        score = 0.0

        # 5-7-5 の段階スコア
        for i in range(3):
            m = self.mora.count_mora(lines[i])
            diff = abs(m - TARGETS_575[i])
            if diff == 0:
                score += 0.24
            elif diff == 1:
                score += 0.08
                strict_575 = False
            elif diff == 2:
                score -= 0.15
                strict_575 = False
            else:
                score -= 0.40
                strict_575 = False
                too_far = True
            score += max(0.0, 0.04 * (1.0 - diff / 3.0))

        # 切れ字
        score += self._kireji_position_bonus(formatted_text)
        kireji_num = count_kireji(raw_text)
        kireji_penalty = 0.0
        if kireji_num == 1:
            score += 0.10
        elif kireji_num > 1:
            kireji_penalty = 0.30

        # 季語が含まれているか
        score += 0.10 if (kigo_hint and self._has_kigo(raw_text, kigo_hint)) else -0.10

        # 助詞開始 / 連続文字
        score += 0.05 if self._check_particles_start_by_lines(formatted_text) else -0.06
        score += 0.05 if self._check_repetition(raw_text) else -0.08

        # 行末助詞ペナルティ
        end_particle_penalty = sum(0.22 for line in lines if line_ends_with_particle(line))

        # 多様性 / 数字 / 名詞偏重
        rep_ratio = self._repetition_ratio(raw_text)
        rep_penalty = max(0.0, (0.75 - rep_ratio)) * 0.25
        num_penalty = self._detect_number_penalty(raw_text)
        noun_penalty = self._noun_ratio_penalty(raw_text)

        score = score - kireji_penalty - end_particle_penalty - rep_penalty - num_penalty - noun_penalty

        # 上限
        if strict_575 and not too_far:
            max_score = 0.97 if (kireji_num == 1 and end_particle_penalty == 0) else 1.0
            if max_score == 0.97:
                score += 0.02
        elif too_far:
            max_score = 0.55
        else:
            max_score = 0.82

        score = max(0.0, min(score, 1.5))
        score = self._soft_cap(score, max_score, strength=6.0)
        return max(0.0, min(score, 1.0))


# ============================================================
# システム（Flask からも呼びやすい）
# ============================================================
class HaikuSystem:
    def __init__(
        self,
        model,
        word_to_idx,
        idx_to_word,
        word_mora,
        mora_counter,
        scorer,
        evaluator,
        device,
        eval_thr: float = 0.5,
    ):
        self.model = model
        self.word_to_idx = word_to_idx
        self.idx_to_word = idx_to_word
        self.word_mora = word_mora
        self.mora_counter = mora_counter
        self.scorer = scorer
        self.evaluator = evaluator
        self.device = device
        self.eval_thr = float(eval_thr)

        self.model.eval()
        self.model.to(self.device)

    def _require_kigo(self, kigo: str) -> str:
        kigo = (kigo or "").strip()
        if not kigo:
            raise ValueError("季語（kigo）は必須です（空は不可）。")
        if kigo in INVALID_KIGO:
            raise ValueError(f"無効な季語です: {kigo}")
        if kigo not in self.word_to_idx:
            raise ValueError(f"季語が語彙にありません: {kigo}")
        return kigo

    def generate(self, kigo: str, num_candidates: int = 300, top_k: int = 10, return_candidates: bool = True) -> dict:
        kigo = self._require_kigo(kigo)
        num_candidates = max(1, min(int(num_candidates), 300))
        top_k = max(1, int(top_k))

        cands = model_candidates(
            self.model, kigo, self.word_to_idx, self.idx_to_word, self.word_mora, self.device, num_candidates=num_candidates
        )
        if not cands:
            raise RuntimeError("候補を生成できませんでした。")

        raws = [r for (_, r) in cands]
        probs = self.evaluator.batch_prob(raws, batch_size=256)
        passed = [(f, r, float(p)) for (f, r), p in zip(cands, probs) if float(p) >= self.eval_thr]

        if not passed:
            return {"ok": False, "kigo": kigo, "reason": "evaluator_reject_all", "eval_thr": self.eval_thr, "generated": len(cands)}

        scored = []
        for f, r, p in passed:
            s = self.scorer.score(f, r, kigo_hint=kigo)
            scored.append((s, f, r, p))
        scored.sort(key=lambda x: x[0], reverse=True)

        best_s, best_f, best_r, best_p = scored[0]
        lines = best_f.split("\n")[:3]
        mora_list = [self.mora_counter.count_mora(x) for x in lines]

        res = {
            "ok": True,
            "kigo": kigo,
            "best": {
                "score": float(best_s),
                "evaluator_prob": float(best_p),
                "haiku": best_f,
                "raw": best_r,
                "lines": lines,
                "mora": mora_list,
            },
            "stats": {"generated": len(cands), "passed_evaluator": len(passed), "eval_thr": self.eval_thr},
        }

        if return_candidates:
            top = scored[:top_k]
            res["top"] = [
                {
                    "score": float(s),
                    "evaluator_prob": float(p),
                    "haiku": f,
                    "raw": r,
                    "mora": [self.mora_counter.count_mora(x) for x in f.split("\n")[:3]],
                }
                for s, f, r, p in top
            ]
        return res

def build_system(
    config_path: str = "data/haiku_config_v2.json",
    evaluator_model_path: str = "model/haiku_evaluator_model.pth",
    evaluator_vocab_path: str = "model/haiku_vocab.json",
    eval_thr: float = 0.77,
) -> HaikuSystem:
    mora = MoraCounter()
    model, w2i, i2w = load_model_and_vocab(config_path=config_path, model_path="model/haiku_generator_best.pt")
    word_mora = build_word_mora_table(w2i, mora)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    evaluator = HaikuEvaluator(model_path=evaluator_model_path, vocab_path=evaluator_vocab_path, max_len=None, device=device)
    scorer = HaikuScorer(mora_counter=mora)

    return HaikuSystem(model, w2i, i2w, word_mora, mora, scorer, evaluator, device, eval_thr=eval_thr)

def main():
    print("=== 季語俳句生成システム（モデル + ルール + モーラ） ===")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"デバイス: {device}")

    try:
        model, w2i, i2w = load_model_and_vocab("data/haiku_config_v2.json", None)
    except Exception as e:
        print(f"致命的エラー：モデル/設定の読み込みに失敗しました: {e}")
        return

    mora = MoraCounter()
    word_mora = build_word_mora_table(w2i, mora)

    evaluator = HaikuEvaluator(
        model_path="model/haiku_evaluator_model.pth",
        vocab_path="model/haiku_vocab.json",
        max_len=None,
        device=device,
    )
    scorer = HaikuScorer(mora_counter=mora)
    eval_thr = 0.77

    model.eval()
    model.to(device)

    while True:
        print("\n操作を選んでください:")
        print("1) 季語を指定して生成（語彙に存在する必要あり）")
        print("2) 終了")
        choice = input("入力 (1-2): ").strip()

        if choice == "2":
            print("終了します。")
            break
        if choice != "1":
            print("入力が正しくありません。")
            continue

        kigo = input("季語を入力してください（語彙に存在するもの）: ").strip()
        if not kigo:
            print("季語が空です。")
            continue
        if kigo in INVALID_KIGO:
            print(f"無効な季語です: {kigo}")
            continue
        if kigo not in w2i:
            print("その季語は語彙にありません。")
            continue

        print(f"\n季語「{kigo}」を含む俳句を生成しています…")

        cands = model_candidates(model, kigo, w2i, i2w, word_mora, device, num_candidates=300)
        if not cands:
            print("候補を生成できませんでした。")
            continue

        raws = [r for (_, r) in cands]
        probs = evaluator.batch_prob(raws, batch_size=256)
        passed = [(f, r, float(p)) for (f, r), p in zip(cands, probs) if float(p) >= eval_thr]

        print(f"候補 {len(cands)} → 評価器通過 {len(passed)}（thr={eval_thr}）")
        if not passed:
            print("評価器が全てを不適切と判断しました（この回はスキップ）。")
            continue

        scored = []
        for f, r, p in passed:
            s = scorer.score(f, r, kigo_hint=kigo)
            scored.append((s, f, r, p))
        scored.sort(key=lambda x: x[0], reverse=True)

        best_s, best_f, best_r, best_p = scored[0]
        lines = best_f.split("\n")[:3]
        mora_list = [mora.count_mora(x) for x in lines]

        print("\n=== 採用結果 ===")
        print(best_f)
        print(f"\nスコア: {best_s:.4f} / evaluator_prob: {best_p:.4f}")
        print(f"モーラ: {mora_list[0]}-{mora_list[1]}-{mora_list[2]}")

        print("\n=== 上位候補（最大10件） ===")
        for i, (s, f, r, p) in enumerate(scored[:10], start=1):
            print(f"\n[{i}] score={s:.4f} / p={p:.4f}")
            print(f)


if __name__ == "__main__":
    main()
