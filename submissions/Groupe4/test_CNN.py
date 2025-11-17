import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import datasets, transforms
from collections import defaultdict
import matplotlib.pyplot as plt
import time
import os

# 
# 1. CONFIGURATION
# 
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DATA_PATH = "data_img"
MAX_IMAGES_PER_CLASS = 2000
BATCH_SIZE = 16
EPOCHS = 20
LEARNING_RATE = 1e-3
NUM_CLASSES = 10
IMG_SIZE = 128

print("="*70)
print(" ANALYSE : RETOURNEMENT HORIZONTAL")
print("="*70)
print(f"Device : {DEVICE}")
print(f"Images/classe : {MAX_IMAGES_PER_CLASS}")
print(f"Epoques : {EPOCHS}")
print("="*70 + "\n")


# 
# 2. TRANSFORMATIONS
# 

# Sans augmentation (baseline)
transform_baseline = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

# Avec retournement horizontal
transform_flip = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])


# 
# 3. CHARGEMENT DES DONNEES
# 

def create_subset_dataset(data_path, transform, max_per_class):
    dataset = datasets.ImageFolder(root=data_path, transform=transform)
    data_loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    
    class_images = defaultdict(list)
    class_labels = defaultdict(list)
    
    for images, labels in data_loader:
        for image, label in zip(images, labels):
            label_idx = label.item()
            if len(class_images[label_idx]) < max_per_class:
                class_images[label_idx].append(image)
                class_labels[label_idx].append(label_idx)
        
        if all(len(class_images[i]) >= max_per_class for i in range(NUM_CLASSES)):
            break
    
    return class_images, class_labels, dataset.classes


class SubsetDataset(Dataset):
    def __init__(self, class_images, class_labels):
        self.images = []
        self.labels = []
        for label, images in class_images.items():
            self.images.extend(images)
            self.labels.extend([label] * len(images))
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        return self.images[idx], self.labels[idx]


# 
# 4. MODELE CNN
# 

class CNN_Animals(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.conv3 = nn.Conv2d(32, 64, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(0.25)
        
        self.fc1 = nn.Linear(64 * 16 * 16, 256)
        self.fc2 = nn.Linear(256, num_classes)
    
    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = x.view(x.size(0), -1)
        x = self.dropout(F.relu(self.fc1(x)))
        x = self.fc2(x)
        return x


# 
# 5. ENTRAINEMENT ET EVALUATION
# 

def train_epoch(loader, model, loss_fn, optimizer):
    model.train()
    total_loss = 0
    for X, y in loader:
        X, y = X.to(DEVICE), y.to(DEVICE)
        pred = model(X)
        loss = loss_fn(pred, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)


def evaluate(loader, model, loss_fn):
    model.eval()
    loss_total = 0
    correct = 0
    size = len(loader.dataset)
    
    with torch.no_grad():
        for X, y in loader:
            X, y = X.to(DEVICE), y.to(DEVICE)
            pred = model(X)
            loss_total += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).sum().item()
    
    return loss_total / len(loader), 100 * correct / size


def train_and_evaluate(method_name, transform):
    print("\n" + "="*70)
    print(f"{method_name}")
    print("="*70)
    
    start_time = time.time()
    
    class_images, class_labels, class_names = create_subset_dataset(
        DATA_PATH, transform, MAX_IMAGES_PER_CLASS
    )
    subset_dataset = SubsetDataset(class_images, class_labels)
    
    total_images = len(subset_dataset)
    print(f"{total_images} images chargees")
    
    train_size = int(0.8 * total_images)
    valid_size = int(0.1 * total_images)
    test_size = total_images - train_size - valid_size
    
    train_dataset, valid_dataset, test_dataset = random_split(
        subset_dataset, [train_size, valid_size, test_size]
    )
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    print(f"Train: {train_size} | Validation: {valid_size} | Test: {test_size}\n")
    
    model = CNN_Animals(num_classes=NUM_CLASSES).to(DEVICE)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    history = {
        "train_loss": [],
        "valid_loss": [],
        "valid_acc": []
    }
    
    for epoch in range(EPOCHS):
        train_loss = train_epoch(train_loader, model, loss_fn, optimizer)
        valid_loss, valid_acc = evaluate(valid_loader, model, loss_fn)
        
        history["train_loss"].append(train_loss)
        history["valid_loss"].append(valid_loss)
        history["valid_acc"].append(valid_acc)
        
        print(f"Epoque {epoch+1}/{EPOCHS} | "
              f"Train Loss: {train_loss:.4f} | "
              f"Valid Loss: {valid_loss:.4f} | "
              f"Valid Acc: {valid_acc:.2f}%")
    
    test_loss, test_acc = evaluate(test_loader, model, loss_fn)
    
    print(f"\nTest Loss: {test_loss:.4f} | Test Acc: {test_acc:.2f}%")
    
    elapsed_time = time.time() - start_time
    
    return history, elapsed_time, test_acc


# 
# 6. COMPARAISON ET VISUALISATION
# 

def plot_comparison(baseline_history, flip_history, baseline_acc, flip_acc):
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    epochs = range(1, EPOCHS + 1)
    
    axes[0].plot(epochs, baseline_history["train_loss"], label="Baseline", linewidth=2)
    axes[0].plot(epochs, flip_history["train_loss"], label="Flip", linewidth=2)
    axes[0].set_title("Train Loss")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()
    
    axes[1].plot(epochs, baseline_history["valid_loss"], label="Baseline", linewidth=2)
    axes[1].plot(epochs, flip_history["valid_loss"], label="Flip", linewidth=2)
    axes[1].set_title("Validation Loss")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()
    
    axes[2].plot(epochs, baseline_history["valid_acc"], label=f"Baseline ({baseline_acc:.1f}%)", linewidth=2)
    axes[2].plot(epochs, flip_history["valid_acc"], label=f"Flip ({flip_acc:.1f}%)", linewidth=2)
    axes[2].set_title("Validation Accuracy")
    axes[2].grid(True, alpha=0.3)
    axes[2].legend()
    
    plt.tight_layout()
    plt.savefig("flip_augmentation_results.png", dpi=300, bbox_inches="tight")
    plt.show()


def compare_methods():
    baseline_history, baseline_time, baseline_acc = train_and_evaluate(
        "Baseline (sans augmentation)", transform_baseline
    )
    
    flip_history, flip_time, flip_acc = train_and_evaluate(
        "Avec retournement horizontal", transform_flip
    )
    
    print("\n" + "="*70)
    print("RESUME COMPARATIF")
    print("="*70)
    print(f"{'Baseline':<30} {baseline_acc:>6.2f}%  {baseline_time:>6.1f}s")
    print(f"{'Retournement horizontal':<30} {flip_acc:>6.2f}%  {flip_time:>6.1f}s")
    
    improvement = flip_acc - baseline_acc
    
    print("\n" + "="*70)
    if improvement > 0:
        print(f"Amelioration : +{improvement:.2f}%")
    elif improvement < 0:
        print(f"Degradation : {improvement:.2f}%")
    else:
        print("Pas de difference significative")
    print("="*70)
    
    plot_comparison(baseline_history, flip_history, baseline_acc, flip_acc)
    
    return {
        "baseline": {"history": baseline_history, "accuracy": baseline_acc, "time": baseline_time},
        "flip": {"history": flip_history, "accuracy": flip_acc, "time": flip_time},
        "improvement": improvement
    }


# 
# 7. LANCEMENT
# 

if __name__ == "__main__":
    if not os.path.exists(DATA_PATH):
        print(f"Erreur: dossier {DATA_PATH} introuvable.")
        exit(1)
    
    results = compare_methods()
    print("\nAnalyse terminee. Résultats enregistrés.")
