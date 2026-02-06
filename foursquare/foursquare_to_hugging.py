import pandas as pd
import numpy as np
from datasets import Dataset, DatasetDict
from sklearn.preprocessing import LabelEncoder
from huggingface_hub import login
import os

# Faça login no Hugging Face (obtenha seu token de acesso na página do Hugging Face)
hugging_token = os.getenv("HUGGING")
version = "balanced"
login(token=hugging_token)

import pandas as pd
import numpy as np
from datasets import Dataset, DatasetDict
from sklearn.preprocessing import LabelEncoder
from huggingface_hub import login

# ==============================
# CONFIG
# ==============================

CSV_US = "/media/gustavo/e5069d0b-8c3d-4850-8e10-319862c4c082/home/claudio/Downloads/dataset_TIST2015/dataset_tist2015_United_States.csv"
CATEGORY_CSV = "/home/gustavo/Downloads/personalization-apis-movement-sdk-categories.csv"

HF_REPO = "claudiogsc/foursquare-us-sequences-highlevel"

SEQ_LEN = 20
MIN_CHECKINS = 60
TRAIN_RATIO = 0.8
SEED = 42

MAX_ALLOWED_CLASSES = 20   # 🔥 ASSERT CANÔNICO

# ==============================
# TAXONOMIA OFICIAL
# ==============================

def load_category_mapping(csv_path):
    df = pd.read_csv(csv_path)

    required = {"Category Name", "Category Label"}
    if not required.issubset(df.columns):
        raise RuntimeError(
            f"CSV inválido. Esperado {required}, encontrado {set(df.columns)}"
        )

    mapping = {}

    for _, row in df.iterrows():
        fine = row["Category Name"]
        label = row["Category Label"]

        if not isinstance(fine, str) or not isinstance(label, str):
            continue

        # detecta separador hierárquico
        if ">" in label:
            high = label.split(">")[0]
        elif "," in label:
            high = label.split(",")[0]
        else:
            # fallback: label já é o high-level
            high = label

        mapping[fine.strip()] = high.strip()

    print(f"✔ Categorias mapeadas: {len(mapping)}")
    print(f"✔ High-level categories detectadas: {set(mapping.values())}")

    return mapping

# ==============================
# LOAD DATASET
# ==============================

def load_foursquare_us(csv_path):
    df = pd.read_csv(csv_path)
    df["utc_time"] = pd.to_datetime(df["utc_time"])
    df = df.sort_values(["user_id", "utc_time"])
    return df

# ==============================
# SEQUENCES
# ==============================

def build_sequences(df, seq_len, min_checkins):
    samples = []

    for user_id, g in df.groupby("user_id"):
        if len(g) < min_checkins:
            continue

        cats = g["category_id"].values
        hours = g["local_hour"].values
        days = g["local_dayofweek"].values

        for i in range(len(cats) - seq_len):
            samples.append({
                "user_id": int(user_id),
                "cat_seq": cats[i:i+seq_len].tolist(),
                "hour_seq": hours[i:i+seq_len].tolist(),
                "day_seq": days[i:i+seq_len].tolist(),
                "target": int(cats[i+seq_len])
            })

    return samples

def split_train_val(samples, ratio, seed):
    rng = np.random.default_rng(seed)
    rng.shuffle(samples)
    split = int(len(samples) * ratio)
    return samples[:split], samples[split:]

# ==============================
# MAIN
# ==============================

def main():
    login()  # usa HF_TOKEN do ambiente

    print("🔎 Carregando taxonomia oficial...")
    category_map = load_category_mapping(CATEGORY_CSV)

    print("📥 Carregando dataset Foursquare...")
    df = load_foursquare_us(CSV_US)

    print("🧭 Aplicando mapeamento hierárquico...")
    df["category_high"] = df["category"].map(category_map)
    df = df[df["category_high"].notna()]

    print("📊 Distribuição high-level:")
    print(df["category_high"].value_counts())

    num_high = df["category_high"].nunique()
    print(f"✔ Número de categorias high-level: {num_high}")

    if num_high > MAX_ALLOWED_CLASSES:
        raise RuntimeError(
            f"❌ ERRO FATAL: {num_high} categorias detectadas "
            f"(esperado ≤ {MAX_ALLOWED_CLASSES})"
        )

    print("🔢 Codificando categorias...")
    le = LabelEncoder()
    df["category_id"] = le.fit_transform(df["category_high"])

    print("✔ Classes finais:", list(le.classes_))

    print("🧩 Gerando sequências...")
    samples = build_sequences(df, SEQ_LEN, MIN_CHECKINS)

    print(f"✔ Total de sequências: {len(samples)}")

    train_samples, val_samples = split_train_val(
        samples, TRAIN_RATIO, SEED
    )

    dataset = DatasetDict({
        "train": Dataset.from_list(train_samples),
        "validation": Dataset.from_list(val_samples),
    }).with_format("torch")

    print("🚀 Enviando para o Hugging Face Hub...")
    dataset.push_to_hub(HF_REPO, private=False)

    print("✅ Dataset publicado:", HF_REPO)

if __name__ == "__main__":
    main()
