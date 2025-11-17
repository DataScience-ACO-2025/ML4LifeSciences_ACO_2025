import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from encoding_2 import ProteinDataset, SS_TO_INT
from model_2 import ProteinCNN, INPUT_DIM, OUTPUT_DIM

# Définition des Constantes
SS_TO_INT = {'H': 0, 'E': 1, 'C': 2, '<pad>': 3}
OUTPUT_DIM = len(SS_TO_INT) # 4
PAD_INDEX = SS_TO_INT['<pad>'] 
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ==============================================================================
# --- 1. Fonction d'Entraînement ---
# ==============================================================================

def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs):
    """
    Boucle d'entraînement et d'évaluation du modèle.
    """
    model.to(DEVICE)
    best_val_loss = float('inf')

    for epoch in range(num_epochs):

        # --- PHASE D'ENTRAÎNEMENT ---
        model.train()
        train_loss = 0.0
        train_correct_predictions = 0
        train_total_residues = 0

        for X_batch, Y_batch, L_batch in train_loader:
            
            X_batch, Y_batch = X_batch.to(DEVICE), Y_batch.to(DEVICE)
            
            optimizer.zero_grad()
            outputs = model(X_batch)

            loss = criterion(
                outputs.contiguous().view(-1, OUTPUT_DIM), 
                Y_batch.contiguous().view(-1)
            )

            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * X_batch.size(0)

            _, predicted = torch.max(outputs.data, 2)
            mask = (Y_batch != PAD_INDEX)
            
            train_total_residues += mask.sum().item()
            train_correct_predictions += ((predicted == Y_batch) & mask).sum().item()


        train_loss /= len(train_loader.dataset)
        train_accuracy = (train_correct_predictions / train_total_residues) * 100 if train_total_residues > 0 else 0

        # --- PHASE DE VALIDATION ---
        model.eval()
        val_loss = 0.0
        val_correct_predictions = 0
        val_total_residues = 0

        with torch.no_grad():
            for X_batch, Y_batch, L_batch in val_loader:
                X_batch, Y_batch = X_batch.to(DEVICE), Y_batch.to(DEVICE)
                
                outputs = model(X_batch)
                
                loss = criterion(
                    outputs.contiguous().view(-1, OUTPUT_DIM), 
                    Y_batch.contiguous().view(-1)
                )
                val_loss += loss.item() * X_batch.size(0)

                _, predicted = torch.max(outputs.data, 2)
                mask = (Y_batch != PAD_INDEX)
                
                val_total_residues += mask.sum().item()
                val_correct_predictions += ((predicted == Y_batch) & mask).sum().item()

        val_loss /= len(val_loader.dataset)
        val_accuracy = (val_correct_predictions / val_total_residues) * 100 if val_total_residues > 0 else 0

        print(f'Epoch {epoch+1}/{num_epochs} | '
              f'Train Loss: {train_loss:.4f} | Train Acc (Q3): {train_accuracy:.2f}% | '
              f'Val Loss: {val_loss:.4f} | Val Acc (Q3): {val_accuracy:.2f}%')

        # Sauvegarde du meilleur modèle
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), 'best_model.pth')
            print("Modèle sauvegardé (meilleure perte de validation)")


# ==============================================================================
# --- Exemple ---
# ==============================================================================

if __name__ == '__main__':

    TRAIN_FILE = '/Users/constancebeaufils/Documents/Master 2/Projet ML-CS/data_pre_processing/training_30_dataset.json'  
    VAL_FILE = '/Users/constancebeaufils/Documents/Master 2/Projet ML-CS/data_pre_processing/validation_dataset.json'

    BATCH_SIZE = 32
    LEARNING_RATE = 0.001
    NUM_EPOCHS = 20 
    
    temp_train_dataset = ProteinDataset(TRAIN_FILE)
    
    L_MAX = temp_train_dataset.max_len
    
    if L_MAX == 0:
        print("Erreur: Le jeu de données est vide. Impossible de continuer l'entraînement.")
    else:
        print(f"L_max : {L_MAX}")
        
        train_dataset = ProteinDataset(TRAIN_FILE, max_len=L_MAX) 
        
        val_dataset = ProteinDataset(VAL_FILE, max_len=L_MAX)
        
        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

        model = ProteinCNN()

        criterion = nn.CrossEntropyLoss(ignore_index=PAD_INDEX) 
        
        optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

        print(f"\nDébut de l'entraînement sur le device: {DEVICE}")
        train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=NUM_EPOCHS)