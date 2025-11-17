# -*- coding: utf-8 -*-
"""
Created on Sun Nov 16 22:36:48 2025

@author: erwan
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torchvision import transforms
from torchvision import datasets
from torch.utils.data import Subset
from torch.utils.data import DataLoader
import PIL
import timm
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, random_split
from sklearn.model_selection import train_test_split
import numpy as np


#Definition de la racine du projet
#Ce chemin doit être modifié sur l'ordinateur de la personne qui l'utilise
os.chdir("C:/Users/erwan/Documents/M2+/BigDataLC/Projet Erwan-Marie-Sixtine")

# Transformation des images pour être lisibles par Vit16 224
train_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
])

test_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
])

#####

#On veut détecter l'espèce de la feuille
#Le code ci-dessous est presque le même que le précendent, à la différence des définitions des jeux d'entrainements et de test

#####
#On charge les images depuis les dossiers train et test
trainesp_dataset = datasets.ImageFolder("trainesp", transform=train_transform)
testesp_dataset = datasets.ImageFolder("testesp", transform=test_transform)

targets = [sample[1] for sample in trainesp_dataset]
train_idx, val_idx = train_test_split(
    range(len(trainesp_dataset)),
    test_size=0.2,
    stratify=targets,
    random_state=42
)

trainesp_strat = Subset(trainesp_dataset, train_idx)
valesp_strat = Subset(trainesp_dataset, val_idx)

trainesp_loader = DataLoader(trainesp_strat, batch_size=32, shuffle=True)
valesp_loader = DataLoader(valesp_strat, batch_size=32, shuffle=False)
testesp_loader = DataLoader(testesp_dataset, batch_size=32, shuffle=True)


#On regarde le nombre de classes que l'on a, ici toutes celles présentent mais on peut également choisir à la main
classes_voulues = trainesp_dataset.classes
num_classes = len(trainesp_dataset.classes)

#On importe le modèle "Vit patch16 224" du package timm, on précise le nombre de neurones sur la couche de sortie pour bien avoir une classification selon notre nombre de classes.
#Par défaut il y a 1000 classes
model = timm.create_model('vit_base_patch16_224',pretrained=True,num_classes=num_classes)

#Si présence de cuda on charge sur le GPU
#Si vous n'avez pas de GPU, il est conseillé de le faire une machine en disposant ou à défaut sur google colab
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

#On définit notre loss et notre optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-4)


#On freeze les paramètres du modèles (environ 81 millions), puis on garde les 3 dernières couches comme entrainable
for param in model.parameters():
    param.requires_grad = False

for param in model.blocks[-1].parameters():
    param.requires_grad = True

for param in model.head.parameters():
    param.requires_grad = True

num_epochs = 10
train_losses = []
val_losses = []
val_accuracies = []   
best_val_loss = 1000
trigger_times = 0
patience = 3

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
        
    for images, labels in trainesp_loader:
        images, labels = images.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        
    avg_train_loss = running_loss / len(trainesp_loader)
    train_losses.append(avg_train_loss)
    print(f"Époque [{epoch+1}/{num_epochs}] - Loss: {avg_train_loss:.4f}")
    
    model.eval()
    running_val_loss = 0.0
    correct = 0
    total = 0
#Validation
    with torch.no_grad():
        for images, labels in valesp_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            running_val_loss += loss.item()

            # Calcul accuracy
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    avg_val_loss = running_val_loss / len(valesp_loader)
    val_losses.append(avg_val_loss)
    val_accuracy = 100 * correct / total
    val_accuracies.append(val_accuracy)

    print(f"Epoch [{epoch+1}/{num_epochs}] "
          f"Train Loss: {avg_train_loss:.4f} "
          f"Val Loss: {avg_val_loss:.4f} "
          f"Val Acc: {val_accuracy:.2f}%")
    
    if avg_val_loss < best_val_loss:
        # amélioration : sauver checkpoint complet (meilleur modèle)
        best_val_loss = avg_val_loss
        trigger_times = 0
        print(f"Validation loss decreased -> saving checkpoint (val_loss={avg_val_loss:.4f})")
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'best_val_loss': best_val_loss
        }, "best-model-espece.pt")
    else:
        trigger_times += 1
        print(f"Pas d'amélioration de la val_loss ({trigger_times}/{patience})")
        if trigger_times >= patience:
            print(f"Early stopping activé à l'époque {epoch+1}")
            break

checkpoint = torch.load("best-model-espece.pt")
model.load_state_dict(checkpoint['model_state_dict']) 

model.eval()

correct = 0
total = 0
        
all_labels = []
all_preds = []
        
with torch.no_grad():
    for images, labels in testesp_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                all_labels.extend(labels.cpu().numpy())
                all_preds.extend(predicted.cpu().numpy())
        
accuracy = 100 * correct / total
print(f"Précision sur le test set: {accuracy:.2f}%")
cm = confusion_matrix(all_labels, all_preds)
print(cm)
        
class_names = testesp_loader.dataset.classes
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
disp.plot(cmap=plt.cm.Blues)
plt.xlabel("Predicted label", fontsize=14,fontweight="bold")
plt.ylabel("True label", fontsize=14,fontweight="bold")
plt.title("Matrice de confusion - Détection d'espèces",fontsize = 16,fontweight="bold",pad=15)
plt.suptitle(f"Test Accuracy: {accuracy:.2f}%", fontsize=14, fontweight="bold", y = 0.93)
plt.xticks(rotation=45, ha="right",fontsize = 12)
plt.yticks(fontsize = 12)
plt.show()

#On regarde l'epoch à laquelle on a sauvegarder notre modèle#
best_epoch = np.argmin(val_losses)
best_val_loss = val_losses[best_epoch]

plt.figure(figsize=(8, 5))
plt.plot(train_losses, label="Train Loss")
plt.plot(val_losses, label="Validation Loss")
plt.xlabel("Épochs", fontsize=14,fontweight="bold")
plt.ylabel("Loss", fontsize=14,fontweight="bold")
plt.axvline(x=best_epoch, color='k', linestyle='--', label='Modèle sélectionné')
plt.text(best_epoch + 0.1, best_val_loss + 0.01, f'Min Val Loss\nEpoch {best_epoch+1}', color='k', fontsize=12, fontweight='bold')
plt.title("Courbe de train loss - Detection d'espèces",fontsize = 16,fontweight="bold",pad=15)
plt.legend()
plt.grid(True)
plt.show()