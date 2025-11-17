# -*- coding: utf-8 -*-
"""
Created on Mon Nov 17 13:53:21 2025

@author: melin
"""

""" Classification d'images avec PyTorch

Concernant le chargement des données du code, il va falloir créer un dossier pomme dans vos documents. 
Le dossier pomme doit contenir deux dossiers "mures" et "pourries" contenant les photos correspondantes.

Chargement des packages : nécessite l'installation préalable de torch
"""

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import time

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
import matplotlib.pyplot as plt


path = "C:\\Users\\melin\\Documents\\Pomme" # à modifier en fonction de vos fichiers

# Vérification du nombre d'images
extensions = (".jpg", ".jpeg", ".png", ".bmp", ".gif")

count = sum(
    len([f for f in files if f.lower().endswith(extensions)])
    for _, _, files in os.walk(path)
)

print("Nombre total d'images :", count)


"""Chargement des données. 
Transformation des données en teseurs 
puis séparation des données en train et test."""


transform = transforms.Compose([
    transforms.Resize((128,128)),
    transforms.ToTensor()
])

full_dataset = datasets.ImageFolder(root=path, transform=transform)

val_size = int(0.2 * len(full_dataset))
train_size = len(full_dataset) - val_size
train_dataset, val_dataset = random_split(
    full_dataset,
    [train_size, val_size],
    generator=torch.Generator().manual_seed(123)
)

train_loader = DataLoader(train_dataset, batch_size=100, shuffle=True)
val_loader   = DataLoader(val_dataset,   batch_size=100, shuffle=False)

"""Construction du modèle"""
class AppleCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(32 * 30 * 30, 64)
        self.fc2 = nn.Linear(64, 1)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, start_dim=1)
        x = F.relu(self.fc1(x))
        x = torch.sigmoid(self.fc2(x))
        return x

model = AppleCNN()

criterion = nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

train_acc_list, val_acc_list = [], []
train_loss_list, val_loss_list = [], []

"""Entrainement et validation de notre modèle"""

start = time.time()

for epoch in range(10):
    # ------- TRAIN -------
    model.train()
    running_loss = 0.0
    running_correct = 0
    total_train = 0

    for images, labels in train_loader:
        images = images
        labels = labels.float().view(-1, 1)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        preds = (outputs >= 0.5).float()
        running_correct += (preds == labels).sum().item()
        total_train += labels.size(0)

    epoch_loss = running_loss / total_train
    epoch_acc = running_correct / total_train

    # ------- VALIDATION -------
    model.eval()
    val_loss = 0.0
    val_correct = 0
    total_val = 0

    with torch.no_grad():
        for images, labels in val_loader:
            images = images
            labels = labels.float().view(-1, 1)

            outputs = model(images)
            loss = criterion(outputs, labels)

            val_loss += loss.item() * images.size(0)
            preds = (outputs >= 0.5).float()
            val_correct += (preds == labels).sum().item()
            total_val += labels.size(0)

    val_loss /= total_val
    val_acc = val_correct / total_val

    train_loss_list.append(epoch_loss)
    train_acc_list.append(epoch_acc)
    val_loss_list.append(val_loss)
    val_acc_list.append(val_acc)

    print(f"Epoch {epoch+1}/{10} "
          f"- train loss: {epoch_loss:.4f}, acc: {epoch_acc:.4f} "
          f"- val loss: {val_loss:.4f}, acc: {val_acc:.4f}")

end = time.time()

print("Durée totale de l'entrainement :", (end-start)/60, "minutes")


"""Pour tracer les graphiques"""

epochs_range = range(10)
plt.figure(figsize=(10, 5))

# Accuracy
plt.subplot(1, 2, 1)
plt.plot(epochs_range, train_acc_list, label='Training Accuracy')
plt.plot(epochs_range, val_acc_list, label='Validation Accuracy')
plt.title('Accuracy')
plt.legend()

# Loss
plt.subplot(1, 2, 2)
plt.plot(epochs_range, train_loss_list, label='Training Loss')
plt.plot(epochs_range, val_loss_list, label='Validation Loss')
plt.title('Loss')
plt.legend()

plt.show()