import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision import datasets
from torch.utils.data import DataLoader
import torch.optim as optim



start = torch.cuda.Event(enable_timing=True)
end = torch.cuda.Event(enable_timing=True)

start.record()

!nvidia-smi # permet de voir si le gpu est bien utilisé mais j'arrive pas a trouver ou voir l'info mdr

# OPTIMISATIONS PYTORCH
torch.backends.cudnn.benchmark = True  # Accélère les convolutions pour des tailles d’images fixes
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("Utilisation de :", device)

end.record()
# Waits for everything to finish running
torch.cuda.synchronize()
print(f"{start.elapsed_time(end):.2f}ms")


##### Implementation de AlexNet #####

## Creation de l'architecture 
# -Feature Extractor: La première couche de convolution prend une image à 3 canaux et passe un kernel de taille 11x11 avec un pas de 4 réduit. Les fonction d'activation sont des ReLu (conserve les y positives et renvoi 0 pour les y négatifs) Ensuite, un maxpooling est appliqué pour réduire la dimension des images et concerver les informations importantes
# -Classifier: Ensemble de percptron multi-couches avec des fonction d'activation ReLu Les features (les x) sont projetées dans un espace de grande dimension ici de taille 4096. La dernière couches est la sortie du reseau
# -Forward: Calcul du lien entre les entrées et les sorties (f(x)) ou on défini le passage des entrées dans le Feature extractor, les features sont transformées en vecteur (fletten) puis les vecteurs de features sont passées dans les PMC.

  class AlexNet(nn.Module):
      def __init__(self, num_classes=1000): #num_classes a modifier en fonction du nombre de sorties souhaité
          super(AlexNet, self).__init__()

          # Feature Extractor = couches de convolution et pooling
          self.features = nn.Sequential(
              nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),  # Conv1
              nn.ReLU(inplace=True),
              nn.MaxPool2d(kernel_size=3, stride=2),  # MaxPool1
              nn.Conv2d(64, 192, kernel_size=5, padding=2),  # Conv2
              nn.ReLU(inplace=True),
              nn.MaxPool2d(kernel_size=3, stride=2),  # MaxPool2
              nn.Conv2d(192, 384, kernel_size=3, padding=1),  # Conv3
              nn.ReLU(inplace=True),
              nn.Conv2d(384, 256, kernel_size=3, padding=1),  # Conv4
              nn.ReLU(inplace=True),
              nn.Conv2d(256, 256, kernel_size=3, padding=1),  # Conv5
              nn.ReLU(inplace=True),
              nn.MaxPool2d(kernel_size=3, stride=2),  # MaxPool3
          )

          # Classifier = couches de PMC
          self.classifier = nn.Sequential(
              nn.Dropout(),
              nn.Linear(256 * 6 * 6, 4096),
              nn.ReLU(inplace=True),
              nn.Dropout(),
              nn.Linear(4096, 4096),
              nn.ReLU(inplace=True),
              nn.Linear(4096, num_classes),  #Couche de sortie
          )

      #Calcul des f(x)
      def forward(self, x):
          x = self.features(x)  # Calcul a traver les couches de convolution
          x = torch.flatten(x, 1)  # Les sorties de la convolution sont transformées en vecteurs pour être utilisable par les PMC
          x = self.classifier(x)  # Calcul a traver les couches de PMC
          return x
      

## Traitement des données
# -Transformation des données:
# [ Ici la taille des images est modifiée, les images sont converties en tenseurs et enfin normalisées. En effet, AlexNet doit traiter des images 
# en 227x227 pixels. Le normalisation permet une convergence plus rapide. ]
# -Chargement des données
# -Création d'un dataloader

from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split

# Transformation des données
transform = transforms.Compose([
    transforms.Resize((227, 227)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

# Chargement du dataset
root = r"/content/drive/MyDrive/PROJET BAYESIEN/DATA/data_test_bug_ant_bee"

dataset = datasets.ImageFolder(root=root, transform=transform)

# Division en train / test
train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size

train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

# DataLoaders
train_loader = DataLoader(
    train_dataset, batch_size=128, shuffle=True,
    num_workers=0, pin_memory=True, persistent_workers=False
)
test_loader = DataLoader(
    test_dataset, batch_size=128, shuffle=False,
    num_workers=0, pin_memory=True, persistent_workers=False
)

print(f"Nombre d'images total : {len(dataset)}")
print(f"Train : {len(train_dataset)}, Test : {test_size}")
print(f"Classes : {dataset.classes}")


#### Entrainement du modèle 
# -Définition du modèle: On défini notre modèle comme étant l'architecture générée en partie "Creation de l'architecture".
# -Fonction loss: Définition de la fonction loss qui est ici une crossentropy.
# -Optimizeer: Définition de la descente de gradient


# Définition du modèle
model = AlexNet(num_classes=3).to('cuda')  # Move model to GPU

# Fonction loss
criterion = nn.CrossEntropyLoss()

# Optimizer
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=0.0005)

# Scheduler (permet d'optimiser le learning rate à chaque epoch, ici on l'a setup a se mettre a jour tt les 5 epoch)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)



## Boucle d'entrainnement
# -Calcul des y^ avec la fonction forward
# -Mis a jour des poids avec la fonction backward

!nvidia-smi

num_epochs = 10

from tqdm import tqdm  # barre de progression utile
num_epochs = 10

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0

    for images, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}", leave=False):
        try:
            images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)
        except Exception as e:
            print(f"Error loading data for epoch {epoch+1}: {e}")
            continue # Skip this batch and continue with the next

        optimizer.zero_grad(set_to_none=True)
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    scheduler.step()  # on met à jour le learning rate après l’epoch

    avg_loss = running_loss / len(train_loader)
    print(f"Epoch [{epoch+1}/{num_epochs}] - Loss: {avg_loss:.4f}")




## Validation
# Permet de tester la capacité du modèle à généraliser. Ici on ne fait pas l'étape backward car on ne veut pas mettre a jour les poids, 
# mais on réalise uniquement l'étape forward pour réaliser les prédictions

model.eval()  # On déclare qu'on est en validation
total, correct = 0, 0  # Suivi des images bien classées

with torch.no_grad():  # Désactivation du calcul du gradient
    for images, labels in test_loader:
        images, labels = images.to('cuda'), labels.to('cuda')  # Passage des données vers un GPU
        outputs = model(images)  # Forward
        _, predicted = outputs.max(1)  # Donne la prédiction
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

accuracy = 100 * correct / total
print(f'Accuracy: {accuracy:.2f}%')


## Fine-Tuning et transfert d'apprentissage
# En remplaçant la dernière couche entièrement connectée (model.classifier[6]), on indique à AlexNet d'ajuster ses prédictions à notre 
# ensemble de données personnalisé. Les couches antérieures restent intactes, conservant les caractéristiques apprises, tandis que la 
# nouvelle couche apprend à classer vos catégories spécifiques.

from torch import nn
from torchvision.models import alexnet

# Importation du pre-trained AlexNet
model = alexnet(pretrained=True).to('cuda') # Move the model to GPU

# modif de la dernière couche pour avoir que 3 classes + alexnet sur gpu
model.classifier[6] = nn.Linear(4096, 3).to('cuda')

for param in model.features.parameters():
    param.requires_grad = False

# optimizer
optimizer = optim.SGD(model.classifier.parameters(), lr=0.001, momentum=0.9)

# criterion sur GPU
criterion = criterion.to('cuda')

# Boucle train
num_epochs = 5
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0

    for images, labels in train_loader:
        images, labels = images.to('cuda'), labels.to('cuda')

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(train_loader):.4f}")


# Validation
model.eval()  # On déclare qu'on est en validation
total, correct = 0, 0  # Suivi des images bien classées

with torch.no_grad():  # Désactivation du calcul du gradient
    for images, labels in test_loader:
        images, labels = images.to('cuda'), labels.to('cuda')  # Passage des données vers un GPU
        outputs = model(images)  # Forward
        _, predicted = outputs.max(1)  # Donne la prédiction
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

accuracy = 100 * correct / total
print(f'Accuracy: {accuracy:.2f}%')

