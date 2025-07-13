import numpy as np
import pickle
import os
import sys
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sudachipy import dictionary, tokenizer
import pyopenjtalk
import random

# 俳句自動生成
class HaikuGenerator:
    def __init__(self, model_path='model/best_model.keras', word2id_path='dataset/word2id.pkl', id2word_path='dataset/id2word.pkl'):
        # モデルの読み込み
        self.model = load_model(model_path)

        # word2idとid2wordの読み込み
        with open(word2id_path, 'rb') as f:
            self.word2id = pickle.load(f)
        with open(id2word_path, 'rb') as f:
            self.id2word = pickle.load(f)

        # 日本語の形態素解析器を初期化
        self.tokenizer_obj = dictionary.Dictionary(dict="full").create()
        self.split_mode = tokenizer.Tokenizer.SplitMode.C

        # モデルのパラメータを設置
        self.max_input_len = 16
        self.vocab_size = len(self.word2id)

    #　テキストを仮名読みに変換
    def to_kana(self, text):
        try:
            return pyopenjtalk.g2p(text, kana=True)
        except:
            return text
        
    # テキストのモーラ数（音節数）を計算
    def count_mora(self, text):
        kana = self.to_kana(text)
        mora_count = 0
        i = 0
        while i < len(kana):
            char = kana[i]

            # ひらがな・カタカナ以外の文字や特殊記号をスキップ
            if not self._is_kana(char) or char in "ーッ":
                i += 1
                continue

            # 小書き仮名が含まれているかを確認し、含まれていれば直前の文字と結合して1モーラとして扱う
            if i + 1 < len(kana) and kana[i + 1] in 'ャュョァィゥェォ':
                mora_count += 1
                i += 2
            else:
                mora_count += 1
                i += 1
        return mora_count

    # 文字が仮名かどうかを判定
    def _is_kana(self, char):
        return (('\u3040' <= char <= '\u309F') or ('\u30A0' <= char <= '\u30FF'))

    # ニューラルネットワークモデルを使って、シーケンス中の次に最も可能性の高い単語を予測
    def predict_next_word(self, seq_ids, temperature=0.8):

        # シーケンスを固定長にパディング
        padded = pad_sequences([seq_ids], maxlen=self.max_input_len, padding='pre')

        # モデルを用いて確率分布を予測
        probs = self.model.predict(padded, verbose=0)[0]

        # 特殊トークンを除外
        special_tokens = ['<PAD>', '<UNK>', '<START>', '<END>']
        for token in special_tokens:
            if token in self.word2id:
                probs[self.word2id[token]] = 0

        # 温度パラメータを適用
        probs = np.log(probs + 1e-10) / temperature
        probs = np.exp(probs)
        probs = probs / np.sum(probs)

        # 確率が最も高い上位100語を選択
        top_k = min(100, len(probs))
        top_indices = np.argsort(probs)[-top_k:]
        top_probs = probs[top_indices]
        top_probs /= np.sum(top_probs)

        #確率に基づいてランダムに選択
        return np.random.choice(top_indices, p=top_probs)

    def _split_into_lines(self, words):
        """
        単語リストを5-7-5のモーラパターンに従って3行に分割する

        Args:
            words: 単語のリスト

        Returns:
            3行に分割された単語のリスト(俳句の各句に対応)
        """

        mora_targets = [5, 7, 5]
        lines = [[], [], []]
        current_line = 0
        current_mora = 0
        for word in words:
            mora = self.count_mora(word)
            if mora == 0:
                continue

            # 現在の行に追加できるかどうかを確認
            if current_line < 3 and current_mora + mora <= mora_targets[current_line]:
                lines[current_line].append(word)
                current_mora += mora

                # 現在の行のモーラ数がmora_targetsに達した場合、次の行に切り替える
                if current_mora == mora_targets[current_line]:
                    current_line += 1
                    current_mora = 0

        return lines

    # 俳句の生成
    def generate_haiku(self, keyword, trials=50, temperature=0.8):
        """
        俳句を生成する

        Args:
            keyword: キーワード
            trials: 最大試行回数
            temperature: 生成時の温度パラメータ

        Returns:
            3行の俳句のタプル。生成に失敗した場合は (None, None, None) を返す
        """

        # キーワードに対応するIDを検索
        keyword_id = self._find_keyword_id(keyword)
        if keyword_id is None:
            print("キーワードを認識できない")
            return None, None, None

        for trial in range(trials):
            sequence = [keyword_id] # 単語IDのシーケンス
            output = [self.id2word[keyword_id]] # 単語を出力
            total_mora = self.count_mora(self._extract_surface_form(output[0]))
            used_words = {output[0]} # 単語の重複を避ける

            # 17モーラに達するか、単語数の上限に達するまで単語を生成する
            while total_mora < 17 and len(output) < 15:
                next_id = self.predict_next_word(sequence, temperature)
                next_word = self.id2word.get(next_id, "<UNK>")
                surface = self._extract_surface_form(next_word)

                # フィルター：重複単語・特殊トークン・空の単語を避ける
                if next_word in used_words or next_word in ['<PAD>', '<UNK>'] or not surface:
                    continue
                mora = self.count_mora(surface)

                if total_mora + mora > 17 or mora == 0:
                    continue

                # 新しい単語を追加
                output.append(next_word)
                sequence.append(next_id)
                used_words.add(next_word)

                # 17モーラに達したかどうかをチェック
                total_mora += mora
                if total_mora == 17:
                    surfaces = [self._extract_surface_form(w) for w in output]
                    lines = self._split_into_lines(surfaces)

                    #　俳句の構造をチェック
                    if len(lines) == 3 and all(lines):
                        moras = [sum(self.count_mora(w) for w in line) for line in lines]
                        if moras == [5, 7, 5]:
                            return (''.join(lines[0]), ''.join(lines[1]), ''.join(lines[2]))
        print(f'俳句の生成に失敗しました！試行回数：{trials}')
        return None, None, None

    # キーワードに対応するIDを検索
    def _find_keyword_id(self, keyword):

        # 直接マッチング
        if keyword in self.word2id:
            return self.word2id[keyword]
        
        # word2idの中から検索
        for entry in self.word2id:
            parts = entry.split('/')
            if len(parts) >= 3 and (keyword in parts[0] or keyword in parts[2]):
                return self.word2id[entry]
            
        # 日本語形態素解析器で処理
        try:
            tokens = self.tokenizer_obj.tokenize(keyword, self.split_mode)
            for token in tokens:
                cand = f"{token.surface()}/{token.reading_form()}/{token.dictionary_form()}/{token.part_of_speech()[0]}"
                if cand in self.word2id:
                    return self.word2id[cand]
        except:
            pass
        return self.word2id.get("<UNK>")

    # word2idを「/」で分割してから、元の単語を抽出する　
    #　例:　"春/ハル/春/名詞" → "春"
    def _parse_word_entry(self, entry):
        parts = entry.split('/')
        if len(parts) >= 4:
            return parts[0]
        return entry

    # _parse_word_entry() のラップ関数
    def _extract_surface_form(self, entry):
        return self._parse_word_entry(entry)
    

if __name__ == "__main__":
    generator = HaikuGenerator()
    while True:
        keyword = input("キーワードを入力してください（'quit' で終了）：").strip()
        if keyword.lower() == 'quit':
            break
        l1, l2, l3 = generator.generate_haiku(keyword)
        if l1:
            print("\n俳句の出力：")
            print(l1)
            print(l2)
            print(l3)
            print(f"モーラー数を検証: {generator.count_mora(l1)}-{generator.count_mora(l2)}-{generator.count_mora(l3)}")
        else:
            print("俳句の生成に失敗しました。別のキーワードを入力してください。")