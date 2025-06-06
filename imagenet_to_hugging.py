from sklearn.model_selection import train_test_split
import os

import pandas as pd
from datasets import Dataset, DatasetDict, Features, ClassLabel, Image
from PIL.Image import fromarray
import cv2

dataset_dir = "ImageNet10"


# Processar as imagens e labels para salvar em um dataframe
# Aqui assumimos que o dataset já foi extraído corretamente.

def read_gtsrb_dataset(dataset_dir):
    # Caminho para os arquivos
    data = []
    labels = []
    width = 64
    height = 64

    # indexes = []
    #
    #     for class_dir in os.listdir(dataset_dir):
    #         class_path = os.path.join(dataset_dir, class_dir)
    #         if os.path.isdir(class_path):
    #             size = len(os.listdir(class_path))
    #             indexes.append((size, class_dir))
    #
    #     indexes = sorted(indexes, key=lambda x: x[0], reverse=True)
    #     indexes = [i[1] for i in indexes]

    for class_dir in os.listdir(dataset_dir)[:10]:
        class_path = os.path.join(dataset_dir, class_dir)
        if os.path.isdir(class_path):
            for img_file in os.listdir(class_path):
                img_path = os.path.join(class_path, img_file)
                image = cv2.imread(img_path)
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                image = cv2.resize(image, (width, height))
                data.append(image)  # Imagem
                labels.append(class_dir)  # Classe da imagem

    # Criar dataframe com as informações
    return data, labels

print("Reading ImageNet10 dataset...")
path = "ImageNet15/train/"

print("Path to dataset files:", path)
data, labels = read_gtsrb_dataset(path)
print("Ready")

# Dividir em treino e teste
X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, random_state=None)

# Criar um dicionário para armazenar os dados
train_data = {'image': [fromarray(i) for i in X_train], 'label': y_train}
test_data = {'image': [fromarray(i) for i in X_test], 'label': y_test}

unique_labels = pd.unique(labels).tolist()

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


# Salvar no Hugging Face
from huggingface_hub import login

hugging_token = os.getenv("HUGGING")
login(token=hugging_token)

# Definir o repositório onde vamos salvar
dataset_name = "ImageNet-10_household_objects"
dataset.push_to_hub(f"claudiogsc/{dataset_name}")
