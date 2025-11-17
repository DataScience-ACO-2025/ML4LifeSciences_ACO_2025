from torch.utils.data import random_split
from torchvision import datasets, transforms
import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.nn.functional as F
import tqdm as tqdm
from data_import import path


# Transforms de base (communes pour l’instant)
basic_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

full_dataset = datasets.ImageFolder(root=path, transform=basic_transform)

# Proportions du split
train_ratio, val_ratio, test_ratio = 0.7, 0.15, 0.15
n_total = len(full_dataset)
n_train = int(n_total * train_ratio)
n_val   = int(n_total * val_ratio)
n_test  = n_total - n_train - n_val

train_set, val_set, test_set = random_split(
    full_dataset,
    [n_train, n_val, n_test],
    generator=torch.Generator().manual_seed(42)
)

print("Train :", len(train_set))
print("Val   :", len(val_set))
print("Test  :", len(test_set))