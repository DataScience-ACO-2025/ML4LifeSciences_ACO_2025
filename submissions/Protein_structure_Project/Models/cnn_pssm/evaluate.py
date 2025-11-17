import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt 
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report, f1_score, precision_recall_fscore_support, accuracy_score

from Models.cnn_pssm.encoding import ProteinDataset, SS_TO_INT, INPUT_DIM

from model import ProteinCNN

SS_TO_INT = {'H': 0, 'E': 1, 'C': 2, '<pad>': 3}
PAD_INDEX = SS_TO_INT['<pad>'] 
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def evaluate_model(model, test_loader, model_path):
    """
    Charge le modèle entraîné et évalue sa précision Q3 sur le jeu de test.
    """
    # Chargement du meilleur modèle
    try:
        model.load_state_dict(torch.load(model_path))
        print(f"\nModèle chargé depuis: {model_path}")
    except FileNotFoundError:
        print(f"\nErreur: Le fichier modèle '{model_path}' n'a pas été trouvé. Assurez-vous que l'entraînement a bien sauvegardé 'best_model.pth'.")
        return 0.0

    model.to(DEVICE)
    model.eval()
    
    total_correct_predictions = 0
    total_residues = 0

    with torch.no_grad():
        for X_batch, Y_batch, L_batch in test_loader:
            X_batch, Y_batch = X_batch.to(DEVICE), Y_batch.to(DEVICE)
            
            outputs = model(X_batch) 
            
            _, predicted = torch.max(outputs.data, 2)

            mask = (Y_batch != PAD_INDEX)
            
            total_residues += mask.sum().item()
            total_correct_predictions += ((predicted == Y_batch) & mask).sum().item()

    q3_accuracy = (total_correct_predictions / total_residues) * 100 if total_residues > 0 else 0.0
    
    return q3_accuracy, total_correct_predictions, total_residues

def analyze_metrics(all_labels, all_predictions, class_labels):
    """
    Calcule la précision, le rappel, le F1-score et génère les plots de la matrice
    de confusion et des métriques par classe.
    """
    target_names = [label.split(' ')[0] for label in class_labels] # ['H', 'E', 'C']
    
    print("\n" + "="*40)
    print("      RAPPORT DE CLASSIFICATION COMPLET")
    print("="*40)

    print(classification_report(all_labels, all_predictions, labels=[0, 1, 2], target_names=target_names))
    
    # ----------------------------------------------------
    # MATRICE DE CONFUSION (HEATMAP)
    # ----------------------------------------------------
    cm = confusion_matrix(all_labels, all_predictions, labels=[0, 1, 2])
    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] 

    plt.figure(figsize=(7, 6))
    sns.heatmap(cm_norm, annot=cm, fmt='d', cmap='Blues', 
                xticklabels=target_names, yticklabels=target_names)
    plt.ylabel('Classe Observée')
    plt.xlabel('Classe Prédite')
    plt.title('Matrice de Confusion')
    plt.show()

    # ----------------------------------------------------
    # PLOT DES MÉTRIQUES PAR CLASSE
    # ----------------------------------------------------
    
    # Calcul des métriques par classe
    precision, recall, f1, _ = precision_recall_fscore_support(
        all_labels, all_predictions, labels=[0, 1, 2], average=None
    )
    
    df_metrics = pd.DataFrame({
        'Classe': target_names,
        'Precision': precision,
        'Rappel': recall,
        'F1-score': f1
    })

    # Plot
    df_metrics.set_index('Classe').plot(kind='bar', figsize=(7, 6))
    plt.ylim(0, 1) # Les scores sont entre 0 et 1
    plt.title('Précision, Rappel et F1-score par Classe')
    plt.ylabel('Score')
    plt.xticks(rotation=0)
    plt.grid(axis='y', linestyle='--')
    plt.show() 

# Exécution de l'évaluation

if __name__ == '__main__':

    TEST_FILE = 'data/testing_matches_subset_dssp.json'  
    MODEL_PATH = 'Models/cnn_pssm/best_model.pth'

    # Chemin du fichier d'entraînement (utilisé comme référence de taille)
    TRAIN_FILE_REF = 'data/matches_subset_dssp.json'
    temp_train_dataset = ProteinDataset(TRAIN_FILE_REF)
    L_MAX = temp_train_dataset.max_len


    CLASS_LABELS = ['H (Alpha - Helix)', 'E (Beta - Strand)', 'C (Coil)']

    test_dataset = ProteinDataset(TEST_FILE, max_len=L_MAX)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    model = ProteinCNN() 

    q3, correct, total = evaluate_model(model, test_loader, MODEL_PATH)

    print("\n========================================")
    print("      RÉSULTATS DE L'ÉVALUATION FINALE")
    print("========================================")
    print(f"Précision Q3 sur le jeu de test: {q3:.2f}%")
    print(f"Résidus correctement prédits: {correct} / {total}")
    print("========================================\n")

    model.eval() 
    all_labels = []
    all_predictions = []

    with torch.no_grad():
        for X_batch, Y_batch, L_batch in test_loader:
            X_batch, Y_batch = X_batch.to(DEVICE), Y_batch.to(DEVICE)
            
            outputs = model(X_batch) 
            _, predicted = torch.max(outputs.data, 2)

            mask = (Y_batch != PAD_INDEX)
            
            labels_np = Y_batch[mask].cpu().numpy()
            predictions_np = predicted[mask].cpu().numpy()

            all_labels.extend(labels_np)
            all_predictions.extend(predictions_np)

    # --- Analyse Détaillée (Matrice, F1, Rappel) ---
    if len(all_labels) > 0:
        # Lancement de la fonction d'analyse détaillée
        analyze_metrics(all_labels, all_predictions, CLASS_LABELS)
    else:
         print("\nAucune donnée trouvée")
    