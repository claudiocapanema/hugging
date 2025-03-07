from sklearn.model_selection import train_test_split
import os
import json

import pandas as pd
from datasets import Dataset, DatasetDict, Features, ClassLabel, Image, Array2D

dataset_dir = "gowalla_texas_sequences_window=10_overlap=0.7.csv"

def read_gowalla_dataset(dataset_dir):
    # Caminho para os arquivos
    data_array  = []
    df = pd.read_csv(dataset_dir)
    data = df['x'].tolist()
    for i in data:
        data_array.append(json.loads(i))
    labels = df['y'].tolist()

    # Criar dataframe com as informações
    return data_array, labels

print("Reading GTSRB dataset...")
path = dataset_dir

print("Path to dataset files:", path)
data, labels = read_gowalla_dataset(dataset_dir)
print("Ready")

# Dividir em treino e teste
X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, random_state=42)

# Criar um dicionário para armazenar os dados
train_data = {'sequence': [i for i in X_train], 'label': y_train}
test_data = {'sequence': [i for i in X_test], 'label': y_test}

unique_labels = pd.unique(labels).tolist()

# Definir as features do dataset (coluna de sequences e rótulos)
features = Features({
    'sequence': Array2D((10, 4), 'float32'),
    'label': ClassLabel(names=unique_labels)  # Ajuste para suas classes reais
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
dataset_name = "Gowalla-State-of-Texas"
dataset.push_to_hub(f"claudiogsc/{dataset_name}")
