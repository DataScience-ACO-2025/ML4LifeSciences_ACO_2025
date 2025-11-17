import torch
import torch.nn as nn
import torch.nn.functional as F

# ==============================================================================
# 1. Constantes d'Architecture
# ==============================================================================

INPUT_DIM = 41   # Dimensions des caractéristiques d'entrée (21 OHE + 20 PSSM)
OUTPUT_DIM = 4   # Nombre de classes de sortie (H, E, C, <pad>)

# Hyperparamètres

KERNEL_SIZE_1 = 3
KERNEL_SIZE_2 = 7
KERNEL_SIZE_3 = 11

FILTERS_1 = 128
FILTERS_2 = 256
FILTERS_3 = 512
DROPOUT_RATE = 0.5


# ==============================================================================
# 2. Classe du Modèle CNN 1D
# ==============================================================================

class ProteinCNN(nn.Module):
  
    def __init__(self, input_dim=INPUT_DIM, output_dim=OUTPUT_DIM):
        super(ProteinCNN, self).__init__()
        
        # 1. Couche 1: Noyau de taille 3
        self.conv1 = nn.Conv1d(
            in_channels=input_dim,          
            out_channels=FILTERS_1,         
            kernel_size=KERNEL_SIZE_1,      
            padding='same' 
        )
        self.dropout1 = nn.Dropout(DROPOUT_RATE)

        # 2. Couche 2: Noyau de taille 7
        self.conv2 = nn.Conv1d(
            in_channels=FILTERS_1,          
            out_channels=FILTERS_2,         
            kernel_size=KERNEL_SIZE_2,      
            padding='same'
        )
        self.dropout2 = nn.Dropout(DROPOUT_RATE)
        
        # 3. Couche 3: Noyau de taille 11
        self.conv3 = nn.Conv1d(
            in_channels=FILTERS_2,          
            out_channels=FILTERS_3,         
            kernel_size=KERNEL_SIZE_3,      
            padding='same'
        )

        # 4. Couche de Classification Finale
        self.output_layer = nn.Conv1d(in_channels=FILTERS_3, out_channels=output_dim, kernel_size=1)
        
        
    def forward(self, x):
        
        x = x.transpose(1, 2)  
        
        # Couches Convolutives et Dropout
        x = F.relu(self.conv1(x))
        x = self.dropout1(x)
        x = F.relu(self.conv2(x))
        x = self.dropout2(x)
        x = F.relu(self.conv3(x))
        
        # Couche de sortie
        x = self.output_layer(x)
    
        x = x.transpose(1, 2)
        return x

# ==============================================================================
# Exemple
# ==============================================================================

if __name__ == '__main__':
    BATCH_SIZE = 32
    L_MAX = 505
    dummy_input = torch.randn(BATCH_SIZE, L_MAX, INPUT_DIM) 
    
    model = ProteinCNN()
    output = model(dummy_input)
    
    print("--- Modèle CNN 1D ---")
    print(f"Dimension de l'input simulé: {dummy_input.shape}")
    print(f"Dimension de l'output du modèle: {output.shape}")
    print(f" => (Batch={BATCH_SIZE}, L_max={L_MAX}, D_input={INPUT_DIM}, D_output={OUTPUT_DIM})")

    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total de paramètres entraînables: {num_params}")