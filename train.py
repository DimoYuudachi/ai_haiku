import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
sns.set()
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Embedding, Dropout
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

# MatplotlibおよびSeabornで日本語を表示可能にする
from matplotlib import rcParams
rcParams['font.family'] = 'MS Gothic'

# 高解像度なPNGでグラフを出力する
import matplotlib_inline.backend_inline
matplotlib_inline.backend_inline.set_matplotlib_formats('retina')

import tensorflow as tf
print("TensorFlow Version:", tf.__version__)

from flask import Flask
print("Flask導入された")

import pyopenjtalk
import pykakasi
from sudachipy import tokenizer, dictionary
import pyopenjtalk
from collections import defaultdict, Counter
import ast
import pickle

# CSVを読み込む
df = pd.read_csv("dataset/processed_haiku.csv", encoding="utf-8")
print(df.head())
print(df.isnull().sum())

# word2id辞書を準備（出現頻度によるフィルタリングあり）

# すべてのトークンの出現頻度を集計
token_counter = Counter()
for idx, row in df.iterrows():
    tokens = ast.literal_eval(row['Tokens'])
    token_counter.update(tokens)

# 最小出現頻度を設定
min_freq = 2

# word2idを初期化
word2id = defaultdict(lambda: word2id["<UNK>"])
word2id["<PAD>"] = 0
word2id["<UNK>"] = 1
word2id["<START>"] = 2
word2id["<END>"] = 3

# 出現頻度が min_freq 以上の単語のみ追加
for token, freq in token_counter.items():
    if freq >= min_freq:
        word2id[token] = len(word2id)

print(f"語彙数（フィルタリング後）: {len(word2id)}")

# word2id保存
with open("dataset/word2id.pkl", "wb") as f:
    pickle.dump(dict(word2id), f)

# id2word保存
id2word = {v: k for k, v in word2id.items()}
with open("dataset/id2word.pkl", "wb") as f:
    pickle.dump(id2word, f)

# word2id を使って、分かち書きされた日本語の単語を ID に変換
haiku_ids_list = []
error_count = 0

for idx, row in df.iterrows():
    # 分かち書きの結果を順に処理
    tokens = ast.literal_eval(row['Tokens'])

    # ID のシーケンスに変換
    ids = [word2id[token] for token in tokens]
    haiku_ids_list.append(ids)

# 乱数シードの設定
np.random.seed(42)
tf.random.set_seed(42)

# モデルのパラメータ
vocab_size = len(word2id)
embedding_dim = 128      # 词嵌入维度
lstm_units = 128         # LSTM单元数
max_length = 16          # 最大序列长度

# 学習データの準備
def prepare_training_data(haiku_ids_list, max_length):
    X, y = [], []
    
    for haiku_ids in haiku_ids_list:
        # すべての俳句にSTARTとENDのトークンを追加
        sequence = [word2id["<START>"]] + haiku_ids + [word2id["<END>"]]
        
        # スライディングウィンドウでシーケンスを作成
        for i in range(1, len(sequence)):
            input_seq = sequence[:i]
            target = sequence[i]
            
            # 長さが16未満のシーケンスのみを処理
            if len(input_seq) <= max_length:
                X.append(input_seq)
                y.append(target)
    
    return X, y

X, y = prepare_training_data(haiku_ids_list, max_length)

print(f"訓練データ数: {len(X)}")
print(f"目標データ数: {len(y)}")

# データパディング
# シーケンスを同じ長さにパディング
print(f"シーケンスを指定した長さまでパディング {max_length}")
X_padded = pad_sequences(X, maxlen=max_length, padding='pre')
y_array = np.array(y)

print(X_padded.shape)
print(y_array.shape)

# モデルの作成
model = Sequential([
    Embedding(vocab_size, embedding_dim),
    LSTM(lstm_units, return_sequences=True, dropout=0.3),
    LSTM(lstm_units, dropout=0.3),
    Dense(lstm_units, activation='relu'),
    Dropout(0.4),
    Dense(vocab_size, activation='softmax')
])

#　モデルの要約
model.build(input_shape=(None, max_length))
model.summary()

# 学習時の設定
# オプティマイザー : Adam
# 損失関数 : スパースカテゴリカルクロスエントロピー
# メトリック : 正解率(accuracy)
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])


# 早期終了（過学習を防止するため）
early_stop = EarlyStopping(
    monitor='val_loss',
    patience=4,
    restore_best_weights=True,
    verbose=1
)

# モデルの自動保存（検証データで最も良いモデルのみを保存）
model_ckpt = ModelCheckpoint(
    filepath='model/best_model.keras',
    monitor='val_loss',
    save_best_only=True,
    mode='min',
    verbose=1
)

# 学習
epoch = 20
hist = model.fit(X_padded, y_array, epochs=epoch, batch_size=64, validation_split=0.1, callbacks=[early_stop, model_ckpt])


# モデルの保存
model.save("model/final_model.keras")

hist.history

# 損失関数の可視化
actual_epoch = len(hist.history['loss'])

fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
ax.plot(range(1, actual_epoch+1), hist.history['loss'], marker='o', label='train')
ax.plot(range(1, actual_epoch+1), hist.history['val_loss'], marker='s', label='val')

ax.set_xlabel('epoch', fontsize=14)
ax.set_ylabel('loss', fontsize=14)
ax.legend()
plt.tight_layout()
plt.show()

# 正解率の可視化
fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
ax.plot(range(1, actual_epoch+1), hist.history['accuracy'], marker='o', label='train')
ax.plot(range(1, actual_epoch+1), hist.history['val_accuracy'], marker='s', label='val')

ax.set_xlabel('epoch', fontsize=14)
ax.set_ylabel('accuracy', fontsize=14)
ax.legend()
plt.tight_layout()
plt.show()