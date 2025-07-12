import numpy as np
import pickle
import os
import sys
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sudachipy import dictionary, tokenizer
import pyopenjtalk
import random

class HaikuGenerator:
    def __init__(self, model_path='model/best_model.keras', 
                 word2id_path='dataset/word2id.pkl', 
                 id2word_path='dataset/id2word.pkl'):
        self.model = load_model(model_path)
        with open(word2id_path, 'rb') as f:
            self.word2id = pickle.load(f)
        with open(id2word_path, 'rb') as f:
            self.id2word = pickle.load(f)
        self.tokenizer_obj = dictionary.Dictionary(dict="full").create()
        self.split_mode = tokenizer.Tokenizer.SplitMode.C
        self.max_input_len = 16
        self.vocab_size = len(self.word2id)

    def to_kana(self, text):
        try:
            return pyopenjtalk.g2p(text, kana=True)
        except:
            return text

    def count_mora(self, text):
        kana = self.to_kana(text)
        mora_count = 0
        i = 0
        while i < len(kana):
            char = kana[i]
            if not self._is_kana(char) or char in "ーッ":
                i += 1
                continue
            if i + 1 < len(kana) and kana[i + 1] in 'ャュョァィゥェォ':
                mora_count += 1
                i += 2
            else:
                mora_count += 1
                i += 1
        return mora_count

    def _is_kana(self, char):
        return (('\u3040' <= char <= '\u309F') or ('\u30A0' <= char <= '\u30FF'))

    def predict_next_word(self, seq_ids, temperature=0.8):
        padded = pad_sequences([seq_ids], maxlen=self.max_input_len, padding='pre')
        probs = self.model.predict(padded, verbose=0)[0]
        special_tokens = ['<PAD>', '<UNK>', '<START>', '<END>']
        for token in special_tokens:
            if token in self.word2id:
                probs[self.word2id[token]] = 0
        probs = np.log(probs + 1e-10) / temperature
        probs = np.exp(probs)
        probs = probs / np.sum(probs)
        top_k = min(100, len(probs))
        top_indices = np.argsort(probs)[-top_k:]
        top_probs = probs[top_indices]
        top_probs /= np.sum(top_probs)
        return np.random.choice(top_indices, p=top_probs)

    def _split_into_lines(self, words):
        mora_targets = [5, 7, 5]
        lines = [[], [], []]
        current_line = 0
        current_mora = 0
        for word in words:
            mora = self.count_mora(word)
            if mora == 0:
                continue
            if current_line < 3 and current_mora + mora <= mora_targets[current_line]:
                lines[current_line].append(word)
                current_mora += mora
                if current_mora == mora_targets[current_line]:
                    current_line += 1
                    current_mora = 0
        return lines

    def generate_haiku(self, keyword, trials=50, temperature=0.8):
        keyword_id = self._find_keyword_id(keyword)
        if keyword_id is None:
            print("关键词无法识别")
            return None, None, None

        for trial in range(trials):
            sequence = [keyword_id]
            output = [self.id2word[keyword_id]]
            total_mora = self.count_mora(self._extract_surface_form(output[0]))
            used_words = {output[0]}
            while total_mora < 17 and len(output) < 15:
                next_id = self.predict_next_word(sequence, temperature)
                next_word = self.id2word.get(next_id, "<UNK>")
                surface = self._extract_surface_form(next_word)
                if next_word in used_words or next_word in ['<PAD>', '<UNK>'] or not surface:
                    continue
                mora = self.count_mora(surface)
                if total_mora + mora > 17 or mora == 0:
                    continue
                output.append(next_word)
                sequence.append(next_id)
                used_words.add(next_word)
                total_mora += mora
                if total_mora == 17:
                    surfaces = [self._extract_surface_form(w) for w in output]
                    lines = self._split_into_lines(surfaces)
                    if len(lines) == 3 and all(lines):
                        moras = [sum(self.count_mora(w) for w in line) for line in lines]
                        if moras == [5, 7, 5]:
                            return (''.join(lines[0]), ''.join(lines[1]), ''.join(lines[2]))
        return None, None, None

    def _find_keyword_id(self, keyword):
        if keyword in self.word2id:
            return self.word2id[keyword]
        for entry in self.word2id:
            parts = entry.split('/')
            if len(parts) >= 3 and (keyword in parts[0] or keyword in parts[2]):
                return self.word2id[entry]
        try:
            tokens = self.tokenizer_obj.tokenize(keyword, self.split_mode)
            for token in tokens:
                cand = f"{token.surface()}/{token.reading_form()}/{token.dictionary_form()}/{token.part_of_speech()[0]}"
                if cand in self.word2id:
                    return self.word2id[cand]
        except:
            pass
        return self.word2id.get("<UNK>")

    def _parse_word_entry(self, entry):
        parts = entry.split('/')
        if len(parts) >= 4:
            return parts[0]
        return entry

    def _extract_surface_form(self, entry):
        return self._parse_word_entry(entry)
    

if __name__ == "__main__":
    generator = HaikuGenerator()
    while True:
        keyword = input("请输入一个关键词（日语词语，输入 'quit' 退出）：").strip()
        if keyword.lower() == 'quit':
            break
        l1, l2, l3 = generator.generate_haiku(keyword)
        if l1:
            print("\n生成的俳句：")
            print(l1)
            print(l2)
            print(l3)
            print(f"音节数验证: {generator.count_mora(l1)}-{generator.count_mora(l2)}-{generator.count_mora(l3)}")
        else:
            print("生成失败，请尝试其他关键词")