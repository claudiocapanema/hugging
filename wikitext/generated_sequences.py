import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import os
import torch
import json
from collections import Counter
from torch.utils.data import DataLoader

import ast
import pandas as pd
from datasets import Dataset, DatasetDict, Features, ClassLabel, Image, Array2D
from datasets import load_dataset

np.set_printoptions(suppress=True)
np.set_printoptions(formatter={'int': lambda x: str(x).replace(" ", ", ")})

# 1. Carregar e tokenizar o dataset
def tokenize(text_lines):
    text = " <eos> ".join(text_lines).lower()
    tokens = text.split()
    return tokens

words = 30

dataset = load_dataset('wikitext', 'wikitext-2-raw-v1')
full_train_tokens = tokenize(dataset['train']['text'])
valid_tokens_raw = tokenize(dataset['validation']['text'])
test_tokens_raw = tokenize(dataset['test']['text'])

# 2. Construir vocabulário com as 100 palavras mais frequentes
counter = Counter(full_train_tokens)
most_common_words = [word for word, _ in counter.most_common(words)]
vocab = {word: idx for idx, word in enumerate(most_common_words)}
vocab_size = len(vocab)
print(f"Vocabulário: {vocab_size} palavras (as 100 mais frequentes)")

# 3. Filtrar os tokens que estão no vocabulário
def filter_tokens(tokens, vocab):
    filtered = [vocab[t] for t in tokens if t in vocab]
    return filtered

train_tokens = filter_tokens(full_train_tokens, vocab)
valid_tokens = filter_tokens(valid_tokens_raw, vocab)
test_tokens = filter_tokens(test_tokens_raw, vocab)

# 4. Dataset para prever a próxima palavra
def get_x_y(data, seq_len):
    x = []
    y = []
    for i in range(len(data) - seq_len):
        x.append(data[i:i + seq_len])
        y.append(data[i + seq_len])
    return x, y

seq_len = 10
batch_size = 256

x, y = get_x_y(train_tokens, seq_len)
print(len(x), len(y))
print(x[0], "\n", y[0])
print(f"unicas cada {np.unique(y, return_counts=True)}")
pd.DataFrame({"X": x, "Y": y}).to_csv(f"wikitext-Window-{seq_len}-Words-{words}.csv", index=False)

