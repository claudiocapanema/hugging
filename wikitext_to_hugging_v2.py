from datasets import load_dataset
from datasets import Sequence, Value

import numpy as np
from sklearn.model_selection import train_test_split
import os
import torch
import pandas as pd
from datasets import Dataset, DatasetDict, Features, ClassLabel, Image, Array2D

np.set_printoptions(suppress=True)
np.set_printoptions(formatter={'int': lambda x: str(x).replace(" ", ", ")})

def tokenize(text: str):
    # Tokenizador simples (pode trocar por spacy ou transformers se quiser algo mais forte)
    return text.lower().strip().split()

def build_vocab(dataset, min_freq: int = 2):
    freq = {}
    for item in dataset:
        for tok in tokenize(item["text"]):
            freq[tok] = freq.get(tok, 0) + 1

    vocab = {"<unk>": 0, "<pad>": 1}
    for word, count in freq.items():
        if count >= min_freq:
            vocab[word] = len(vocab)
    return vocab

def encode(text, vocab):
    return [vocab.get(tok, vocab["<unk>"]) for tok in tokenize(text)]

def batchify(data, batch_size, device):
    # n_batch = len(data) // batch_size
    # data = data[:n_batch * batch_size]
    data = torch.tensor(data, dtype=torch.long, device=device)
    # data = data.view(batch_size, -1).t().contiguous()
    return data

def get_batch(source, i, bptt):
    seq_len = min(bptt, len(source) - 1 - i)
    data = source[i:i+seq_len]
    target = source[i+1:i+1+seq_len].reshape(-1)
    return data, target

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dataset = load_dataset("wikitext", "wikitext-2-raw-v1")

vocab = build_vocab(dataset["train"])
train_data = [tok for ex in dataset["train"] for tok in encode(ex["text"], vocab)]
val_data = [tok for ex in dataset["validation"] for tok in encode(ex["text"], vocab)]

batch_size, bptt = 20, 35
train_data = batchify(train_data, batch_size, device)
val_data = batchify(val_data, batch_size, device)
train_data = train_data + val_data

# 4. Dataset para prever a próxima palavra
def get_x_y(data, seq_len):
    x = []
    y = []
    for i in range(len(data) - seq_len):
        x.append(data[i:i + seq_len])
        y.append(data[i + seq_len])
    return x, y

seq_len = 1

data, labels = get_x_y(train_data, seq_len)

words = len(vocab)

window = 1

# Dividir em treino e teste
X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, random_state=None)

# Criar um dicionário para armazenar os dados
train_data = {'sequence': [i for i in X_train], 'label': y_train}
test_data = {'sequence': [i for i in X_test], 'label': y_test}

unique_labels = pd.unique(labels).tolist()

print(f"quantidade labels {len(unique_labels)} {unique_labels}")

# unique_labels = [i for i in range(max(pd.unique(labels).tolist()))]
# unique_sub_labels = [i for i in range(max(pd.unique(sub_labels).tolist()))]
# unique_sub_sub_labels = [i for i in range(max(pd.unique(sub_sub_labels).tolist()))]

# Definir as features do dataset (coluna de sequences e rótulos)
features = Features({
    'sequence': Sequence(Value("int32")),
    'label': ClassLabel(names=unique_labels)# Ajuste para suas classes reais
})

# Criar os datasets
train_dataset = Dataset.from_dict(train_data, features=features)
test_dataset = Dataset.from_dict(test_data, features=features)

# Organizar os datasets em um DatasetDict
dataset = DatasetDict({
    'train': train_dataset,
    'test': test_dataset
})


# Salvar no Hugging Face
from huggingface_hub import login

hugging_token = os.getenv("HUGGING")
login(token=hugging_token)

# Definir o repositório onde vamos salvar
dataset_name = f"wikitext-Window-{window}-Words-{words}"
dataset.push_to_hub(f"claudiogsc/{dataset_name}")
