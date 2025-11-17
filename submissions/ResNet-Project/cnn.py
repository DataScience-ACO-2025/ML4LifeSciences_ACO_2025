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


# Data augmentation CNN : 

train_transform_cnn = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.1, contrast=0.1),
    transforms.ToTensor(),
])



# Dataloaders

batch_size = 32

train_loader_cnn = DataLoader(train_set, batch_size=batch_size, shuffle=True)
val_loader_cnn   = DataLoader(val_set, batch_size=batch_size, shuffle=False)
test_loader_cnn  = DataLoader(test_set, batch_size=batch_size, shuffle=False)



class PlainCNN(nn.Module):
    def __init__(self, num_classes=5):
        super().__init__()

        self.conv_block1 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

        self.conv_block2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

        self.conv_block3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

        self.fc_block = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 28 * 28, 256),   # après 3 pools : 224 → 112 → 56 → 28
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        x = self.conv_block1(x)
        x = self.conv_block2(x)
        x = self.conv_block3(x)
        x = self.fc_block(x)
        return x


# possible de remplacer mps par cuda si pc
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

print("Device:", device)


model_cnn = PlainCNN(num_classes=5).to(device)

criterion_cnn = nn.CrossEntropyLoss()

optimizer_cnn = torch.optim.Adam(model_cnn.parameters(), lr=1e-3)



