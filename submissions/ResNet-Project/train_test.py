import torch
import tqdm as tqdm
from resnet import train_loader_resnet, val_loader_resnet, test_loader_resnet, model_res, criterion_res, optimizer_res
from cnn import train_loader_cnn, val_loader_cnn, test_loader_cnn, model_cnn, criterion_cnn, optimizer_cnn

device = torch.device("mps")

def train_one_epoch(model, loader, optimizer, criterion):
    model.train()
    total_loss = 0
    correct = 0

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * images.size(0)
        preds = outputs.argmax(dim=1)
        correct += (preds == labels).sum().item()

    return total_loss / len(loader.dataset), correct / len(loader.dataset)


def validate(model, loader, criterion):
    model.eval()
    total_loss = 0
    correct = 0

    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)

            loss = criterion(outputs, labels)
            total_loss += loss.item() * images.size(0)

            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()

    return total_loss / len(loader.dataset), correct / len(loader.dataset)




num_epochs = 20

# CNN

results_cnn = {
    "train_loss": [],
    "train_acc": [],
    "val_loss": [],
    "val_acc": [],
    "test_loss": None,
    "test_acc": None
}

for epoch in tqdm.tqdm(range(num_epochs), desc="Training CNN"):
    train_loss, train_acc = train_one_epoch(model_cnn, train_loader_cnn, optimizer_cnn, criterion_cnn)
    val_loss, val_acc     = validate(model_cnn, val_loader_cnn, criterion_cnn)

    results_cnn["train_loss"].append(train_loss)
    results_cnn["train_acc"].append(train_acc)
    results_cnn["val_loss"].append(val_loss)
    results_cnn["val_acc"].append(val_acc)

    print(f"Epoch {epoch+1}/{num_epochs}")
    print(f"  Train loss : {train_loss:.4f}  |  acc : {train_acc:.4f}")
    print(f"  Val   loss : {val_loss:.4f}  |  acc : {val_acc:.4f}")


test_loss, test_acc = validate(model_cnn, test_loader_cnn, criterion_cnn)
results_cnn["test_loss"] = test_loss
results_cnn["test_acc"]  = test_acc

print("CNN Test accuracy :", test_acc)


# ResNet

results_res = {
    "train_loss": [],
    "train_acc": [],
    "val_loss": [],
    "val_acc": [],
    "test_loss": None,
    "test_acc": None
}

for epoch in tqdm.tqdm(range(num_epochs), desc="Training ResNet"):
    train_loss, train_acc = train_one_epoch(model_res, train_loader_resnet, optimizer_res, criterion_res)
    val_loss, val_acc     = validate(model_res, val_loader_resnet, criterion_res)

    results_res["train_loss"].append(train_loss)
    results_res["train_acc"].append(train_acc)
    results_res["val_loss"].append(val_loss)
    results_res["val_acc"].append(val_acc)

    print(f"Epoch {epoch+1}/{num_epochs}")
    print(f"  Train loss : {train_loss:.4f}  |  acc : {train_acc:.4f}")
    print(f"  Val   loss : {val_loss:.4f}  |  acc : {val_acc:.4f}")


test_loss, test_acc = validate(model_res, test_loader_resnet, criterion_res)
results_res["test_loss"] = test_loss
results_res["test_acc"]  = test_acc

print("ResNet Test accuracy :", test_acc)

import pickle

# Sauvegarde des résultats CNN
with open("results/results_cnn.pkl", "wb") as f:
    pickle.dump(results_cnn, f)

# Sauvegarde des résultats ResNet
with open("results/results_res.pkl", "wb") as f:
    pickle.dump(results_res, f)