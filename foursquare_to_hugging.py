import pandas as pd
import numpy as np
import torch
import random
from sklearn.preprocessing import LabelEncoder
from datasets import Dataset, DatasetDict
from huggingface_hub import login

SEQ_LEN = 5
TOP_K = 100   # <<< somente os 100 POIs mais visitados

BASE = "/media/gustavo/e5069d0b-8c3d-4850-8e10-319862c4c082/home/claudio/Downloads/dataset_TIST2015/"
FILENAME = "dataset_TIST2015_Checkins.txt"

HF_REPO = f"claudiogsc/foursquare_sequences_len_{SEQ_LEN}_top_{TOP_K}_venues"

# login()  # rode apenas uma vez se não estiver logado

df = pd.read_csv(
    f"{BASE}{FILENAME}",
    sep="\t",
    header=None,
    names=["userid", "venueid", "datetime", "lat", "lng"]
)[["userid", "venueid", "datetime"]].head(6000000)

df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce")
df = df.dropna(subset=["datetime"])

df["weekday"] = df["datetime"].dt.weekday.astype(int)
df = df.sort_values(["userid", "datetime"])

df["delta_t"] = df.groupby("userid")["datetime"].diff().dt.total_seconds() / 3600.0
df["delta_t"] = df["delta_t"].fillna(0.0)

df["delta_t_bin"] = pd.cut(
    df["delta_t"],
    bins=[-1, 0.5, 2, 6, 24, 72, 1e9],
    labels=[0,1,2,3,4,5]
).astype(int)

df["hour"] = df["datetime"].dt.hour.astype(int)

# =====================================================
# 🔥 Selecionar somente os TOP 100 venues mais visitados
# =====================================================
venue_counts = df["venueid"].value_counts()
top_venues = venue_counts.head(TOP_K).index

df = df[df["venueid"].isin(top_venues)]

print("Total registros após top venues:", len(df))
print("Total venues:", df["venueid"].nunique())

# =====================================================
# Split por usuários
# =====================================================
users = df["userid"].unique()
np.random.shuffle(users)

train_users = set(users[:int(0.8 * len(users))])
test_users  = set(users[int(0.8 * len(users)):])

train_df = df[df["userid"].isin(train_users)]
test_df  = df[df["userid"].isin(test_users)]

# =====================================================
# LabelEncoder apenas nos TOP 100 venues
# =====================================================
le_venue = LabelEncoder()
train_df["venue_id_enc"] = le_venue.fit_transform(train_df["venueid"])

test_df = test_df[test_df["venueid"].isin(le_venue.classes_)]
test_df["venue_id_enc"] = le_venue.transform(test_df["venueid"])

num_classes = len(le_venue.classes_)
print("Num classes (venues):", num_classes)

# =====================================================
# Construção das sequências
# =====================================================
def build_sequences(df, seq_len=5):
    sequences = []
    labels = []

    for user_id, user_data in df.groupby("userid"):
        venues = user_data["venue_id_enc"].values.astype(np.int64)
        hours = user_data["hour"].values.astype(np.int64)
        weekdays = user_data["weekday"].values.astype(np.int64)
        deltas = user_data["delta_t_bin"].values.astype(np.int64)

        for i in range(len(venues) - seq_len):
            seq = np.stack([
                venues[i:i+seq_len],
                hours[i:i+seq_len],
                weekdays[i:i+seq_len],
                deltas[i:i+seq_len]
            ], axis=1)

            sequences.append(seq.tolist())
            labels.append(int(venues[i+seq_len]))

    return sequences, labels

train_X, train_y = build_sequences(train_df, SEQ_LEN)
test_X, test_y   = build_sequences(test_df, SEQ_LEN)

print("Train samples:", len(train_X))
print("Test samples:", len(test_X))

train_dataset = Dataset.from_dict({
    "sequence": train_X,
    "label": train_y
})

test_dataset = Dataset.from_dict({
    "sequence": test_X,
    "label": test_y
})

dataset_dict = DatasetDict({
    "train": train_dataset,
    "test": test_dataset
})

print(dataset_dict)

dataset_dict.push_to_hub(HF_REPO)
