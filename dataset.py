import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
from torchvision import datasets
import os
import pandas as pd
from torchvision.io import read_image
from torch.utils.data import Dataset,random_split,Subset
from torchvision.transforms import ToTensor
import random
from typing import Union

batch_size = 60

training_data = datasets.FashionMNIST(root="data",train=True,download=True,transform=ToTensor())
test_data = datasets.FashionMNIST(root="data",train=False,download=True,transform=ToTensor())

labels_map = {
    0: "T-Shirt",
    1: "Trouser",
    2: "Pullover",
    3: "Dress",
    4: "Coat",
    5: "Sandal",
    6: "Shirt",
    7: "Sneaker",
    8: "Bag",
    9: "Ankle Boot",
}
'''
#看图
figure = plt.figure(figsize=(8, 8))
cols, rows = 3, 3
for i in range(1, cols * rows + 1):
    sample_idx = torch.randint(len(training_data), size=(1,)).item()
    img, label = training_data[sample_idx]
    figure.add_subplot(rows, cols, i)
    plt.title(labels_map[label])
    plt.axis("off")
    plt.imshow(img.squeeze(), cmap="gray")
plt.show()
'''
class CustomImageDataset(Dataset):
    def __init__(self, annotations_file, img_dir, transform=None, target_transform=None):
        self.img_labels = pd.read_csv(annotations_file)
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx, 0])
        image = read_image(img_path)
        label = self.img_labels.iloc[idx, 1]
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label


from torch.utils.data import DataLoader

train_dataloader = DataLoader(training_data, batch_size=64, shuffle=True)
test_dataloader = DataLoader(test_data, batch_size=64, shuffle=True)

# Display image and label.
train_features, train_labels = next(iter(train_dataloader))
print(f"Feature batch shape: {train_features.size()}")
print(f"Labels batch shape: {train_labels.size()}")
img = train_features[0].squeeze()
label = train_labels[0]
plt.imshow(img, cmap="gray")
plt.show()
print(f"Label: {label}")


#iid-dataloader
def get_iid_dataloaders_test():

    return()

def generate_iid_dataloaders(num_clients: int):
    dataset = training_data
    size = len(dataset) // num_clients
    iid_datasets = random_split(dataset, [size for _ in range(num_clients)])

    return [DataLoader(_dataset, batch_size=batch_size, shuffle=True) for _dataset in iid_datasets]


def generate_random_dataloader():
    dataset = training_data
    size = len(dataset) // 50
    idxes = torch.randint(0, len(dataset), (size,)).tolist()
    subset = Subset(dataset, idxes)

    return [DataLoader(subset, batch_size=batch_size, shuffle=True)]


def generate_test_dataloader(size: Union[int, None] = None):
    dataset = training_data
    if size is not None:
        idxes = random.sample(range(len(dataset)), size)
        dataset = Subset(dataset, idxes)

    return [DataLoader(dataset, batch_size=batch_size, shuffle=True)]