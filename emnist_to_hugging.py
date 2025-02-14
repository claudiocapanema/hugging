import os

import pandas as pd
from datasets import Dataset, DatasetDict, Features, ClassLabel, Image
from huggingface_hub import login
from emnist import extract_training_samples, extract_test_samples
from PIL.Image import fromarray

# Faça login no Hugging Face (obtenha seu token de acesso na página do Hugging Face)
hugging_token = os.getenv("HUGGING")
version = "balanced"
login(token=hugging_token)
# Carregar os caminhos das imagens e seus rótulos
train_images, train_labels = extract_training_samples(version)

test_images, test_labels = extract_test_samples(version)

# Criar um dicionário para armazenar os dados
train_data = {'image': [fromarray(i) for i in train_images], 'label': train_labels}
test_data = {'image': [fromarray(i) for i in test_images], 'label': test_labels}

unique_labels = pd.unique(train_labels).tolist()

# Definir as features do dataset (coluna de imagens e rótulos)
features = Features({
    'image': Image(),
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

# Fazer o upload para o Hugging Face Hub
dataset.push_to_hub(f"claudiogsc/emnist_{version}", private=False)  # 'private=False' se você quiser público
