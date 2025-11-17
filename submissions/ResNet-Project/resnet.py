from torch.utils.data import random_split
from torchvision import datasets, transforms, models
import torch
import tqdm as tqdm
from torch.utils.data import DataLoader
import torch.nn as nn
from torchvision import models

from data_split import full_dataset
from data_split import n_train, n_val, n_test
from data_import import path

train_transform_resnet = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406], 
        std=[0.229, 0.224, 0.225]
    )
])

test_transform_resnet = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406], 
        std=[0.229, 0.224, 0.225]
    )
])


batch_size = 32

full_dataset_resnet = datasets.ImageFolder(root=path, transform=test_transform_resnet)

train_set_r, val_set_r, test_set_r = random_split(
    full_dataset_resnet,
    [n_train, n_val, n_test],
    generator=torch.Generator().manual_seed(42)
)

train_loader_resnet = DataLoader(train_set_r, batch_size=batch_size, shuffle=True)
val_loader_resnet   = DataLoader(val_set_r, batch_size=batch_size, shuffle=False)
test_loader_resnet  = DataLoader(test_set_r, batch_size=batch_size, shuffle=False)


class ResNet18Gray(nn.Module):
    def __init__(self, num_classes):
        super().__init__()

        # Pre-trained model
        self.model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)

        # Last layer adaptation
        self.model.fc = nn.Linear(512, num_classes)

    def forward(self, x):
        return self.model(x)


device = torch.device("mps")
num_classes = len(full_dataset.class_to_idx)


#model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet18', pretrained=True)

model_res = ResNet18Gray(num_classes).to(device)



criterion_res = nn.CrossEntropyLoss()

optimizer_res = torch.optim.Adam(
    model_res.parameters(),
    lr=1e-4,            # lr weaker for pre-trained models
    weight_decay=1e-4
)

#scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
#    optimizer_res, mode='min', factor=0.5, patience=2
#)

