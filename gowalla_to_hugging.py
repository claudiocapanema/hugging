from sklearn.model_selection import train_test_split
import os
import json

import ast
import pandas as pd
from datasets import Dataset, DatasetDict, Features, ClassLabel, Image, Array2D

window = 4
overlap = 0.5
dataset_dir = f"gowalla/gowalla_checkins_texas_sequences_window_{window}_overlap_{overlap}.csv"

def read_gowalla_dataset(dataset_dir):
    # Caminho para os arquivos
    data_array  = []
    df = pd.read_csv(dataset_dir)
    data = df['X'].tolist()
    for i in data:
        data_array.append(ast.literal_eval(i))
    labels = df['Y_category'].tolist()
    sub_labels = df['Y_sub_category'].tolist()
    sub_sub_labels = df['Y_sub_sub_category'].tolist()

    # Criar dataframe com as informações
    return data_array, labels, sub_labels, sub_sub_labels

print("Reading GTSRB dataset...")
path = dataset_dir

print("Path to dataset files:", path)
data, labels, sub_labels, sub_sub_labels = read_gowalla_dataset(dataset_dir)
print("Ready")

# Dividir em treino e teste
X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, random_state=None)
X_train, X_test, sub_y_train, sub_y_test = train_test_split(data, sub_labels, test_size=0.2, random_state=None)
X_train, X_test, sub_sub_y_train, sub_sub_y_test = train_test_split(data, sub_sub_labels, test_size=0.2, random_state=None)

# Criar um dicionário para armazenar os dados
train_data = {'sequence': [i for i in X_train], 'label': y_train, 'sub_label': sub_y_train, 'sub_sub_label': sub_sub_y_train }
test_data = {'sequence': [i for i in X_test], 'label': y_test, 'sub_label': sub_y_test, 'sub_sub_label': sub_sub_y_test}

unique_labels = pd.unique(labels).tolist()
unique_sub_labels = pd.unique(sub_labels).tolist()
unique_sub_sub_labels = pd.unique(sub_sub_labels).tolist()

print(f"quantidade labels {len(unique_labels)} quantidade sub labels {len(unique_sub_labels)} quantidade sub labels {len(unique_sub_sub_labels)}")
exit()

# unique_labels = [i for i in range(max(pd.unique(labels).tolist()))]
# unique_sub_labels = [i for i in range(max(pd.unique(sub_labels).tolist()))]
# unique_sub_sub_labels = [i for i in range(max(pd.unique(sub_sub_labels).tolist()))]

# Definir as features do dataset (coluna de sequences e rótulos)
features = Features({
    'sequence': Array2D((window, 6), 'int32'),
    'label': ClassLabel(names=unique_labels),  # Ajuste para suas classes reais
    'sub_label': ClassLabel(names=unique_sub_labels),  # Ajuste para suas classes reais
    'sub_sub_label': ClassLabel(names=unique_sub_sub_labels),  # Ajuste para suas classes reais
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
dataset_name = f"Gowalla-State-of-Texas-Window-{window}-overlap-{overlap}"
dataset.push_to_hub(f"claudiogsc/{dataset_name}")
