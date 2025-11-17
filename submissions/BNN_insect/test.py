print("ici")

# === IMPORTS NÉCESSAIRES ===
import torch
print("uuu")
import torch.nn as nn
print("saaaaaah")
import torch.optim as optim
print("pffff")
from tqdm import tqdm

print("tutu")

# Assurez-vous d'importer les modules BBB nécessaires
from PyBCNN.layers.BBB import BBBLinear as BBB_Linear, BBBConv as BBB_Conv
from PyBCNN.layers.BBB_LRT import BBBLinear as BBB_Linear_LRT, BBBConv as BBB_Conv_LRT
from PyBCNN.layers.misc import FlattenLayer, ModuleWrapper
from PyBCNN.models.BayesianModels import BayesianAlexNet



#################################################################################################


from torchvision.models import alexnet

print("la")

# 1. Charger AlexNet pré-entraîné
pretrained_alexnet = alexnet(pretrained=True)

# 2. Créer BBBAlexNet
priors = {'prior_mu': 0, 'prior_sigma': 0.1}
model = BayesianAlexNet(
    outputs=2,
    inputs=3,
    priors=priors,
    layer_type='bbb',
    activation_type='relu'  # Utilisez 'relu' pour matcher AlexNet
)

# 3. Transférer les poids vers les moyennes (mu) des distributions bayésiennes
def transfer_weights_to_bayesian(bayesian_model, pretrained_model):
    """
    Transfère les poids d'un modèle classique vers les moyennes (mu) 
    des distributions bayésiennes
    """
    # Mapping des couches conv
    conv_mapping = {
        'conv1': 0,  # pretrained.features[0] -> bayesian.conv1
        'conv2': 3,  # pretrained.features[3] -> bayesian.conv2
        'conv3': 6,  # pretrained.features[6] -> bayesian.conv3
        'conv4': 8,  # pretrained.features[8] -> bayesian.conv4
        'conv5': 10, # pretrained.features[10] -> bayesian.conv5
    }
    
    # Transférer les couches de convolution
    for bay_name, pre_idx in conv_mapping.items():
        bay_layer = getattr(bayesian_model, bay_name)
        pre_layer = pretrained_model.features[pre_idx]
        
        # Copier weight vers weight_mu
        bay_layer.weight_mu.data.copy_(pre_layer.weight.data)
        if pre_layer.bias is not None:
            bay_layer.bias_mu.data.copy_(pre_layer.bias.data)
        
        print(f"✅ Transféré {bay_name} depuis features[{pre_idx}]")
    
    # Transférer la couche classifier (adapter selon votre nombre de classes)
    # Note: AlexNet a 1000 classes, vous en avez 2
    # On ne transfère donc PAS la dernière couche, juste les features
    
    print("✅ Transfert terminé ! Les moyennes bayésiennes sont initialisées avec AlexNet pré-entraîné")

# Appliquer le transfert
transfer_weights_to_bayesian(model, pretrained_alexnet)

# Maintenant model a ses moyennes initialisées avec les poids d'AlexNet
# Les sigmas restent à leur valeur initiale (petite variance)


#################################################################################################



# === FONCTION LOSS ===
criterion = nn.CrossEntropyLoss()

# === OPTIMIZER ===
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=0.0005)

# === SCHEDULER ===
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)

# === BOUCLE D'ENTRAÎNEMENT BAYÉSIENNE ===
num_epochs = 10
num_batches = len(train_loader)  # Important pour normaliser le KL loss

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    running_nll = 0.0
    running_kl = 0.0

    for images, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}", leave=False):
        try:
            images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)
        except Exception as e:
            print(f"Error loading data for epoch {epoch+1}: {e}")
            continue

        optimizer.zero_grad(set_to_none=True)
        
        # Forward pass
        outputs = model(images)
        
        # Negative Log-Likelihood (NLL) loss
        nll_loss = criterion(outputs, labels)
        
        # KL Divergence loss (terme de régularisation bayésien)
        kl_loss = model.kl_loss()  # Méthode fournie par ModuleWrapper
        
        # Loss totale = NLL + KL/nombre_de_batches
        # Le facteur 1/num_batches permet de pondérer correctement le KL
        loss = nll_loss + (kl_loss / num_batches)
        
        # Backward et mise à jour
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        running_nll += nll_loss.item()
        running_kl += kl_loss.item()

    scheduler.step()

    # Affichage des losses
    avg_loss = running_loss / len(train_loader)
    avg_nll = running_nll / len(train_loader)
    avg_kl = running_kl / len(train_loader)
    
    print(f"Epoch [{epoch+1}/{num_epochs}] - Loss: {avg_loss:.4f} | NLL: {avg_nll:.4f} | KL: {avg_kl:.4f}")

# === VALIDATION BAYÉSIENNE ===
model.eval()
total, correct = 0, 0

with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)
        
        # Pour un réseau bayésien, vous pouvez faire plusieurs prédictions (Monte Carlo)
        # et moyenner les résultats pour avoir une meilleure estimation
        num_samples = 10  # Nombre d'échantillonnages Monte Carlo
        predictions = []
        
        for _ in range(num_samples):
            outputs = model(images)
            predictions.append(torch.softmax(outputs, dim=1))
        
        # Moyenne des prédictions
        avg_predictions = torch.stack(predictions).mean(0)
        _, predicted = avg_predictions.max(1)
        
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

accuracy = 100 * correct / total
print(f'Accuracy: {accuracy:.2f}%')