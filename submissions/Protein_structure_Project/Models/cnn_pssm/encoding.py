import json
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader

# --- 1. Définition des Constantes d'Encodage et de Mappage ---

AA_ALPHABET = 'ACDEFGHIKLMNPQRSTVWY'
AA_TO_INT = {aa: i for i, aa in enumerate(AA_ALPHABET)}
AA_TO_INT['<pad>'] = 20

DSSP_3_MAP = {
    'H': 'H', 'G': 'H', 'I': 'H', 
    'E': 'E', 'B': 'E', 
    'T': 'C', 'S': 'C', 'L': 'C', '-': 'C',
}
SS_3_CLASSES = 'HEC'
SS_TO_INT = {ss: i for i, ss in enumerate(SS_3_CLASSES)}
SS_TO_INT['<pad>'] = 3

OHE_DIM = len(AA_ALPHABET) + 1  # 21 (20 AA + <pad>)
PSSM_DIM = 20                   # 20 dimensions PSSM
INPUT_DIM = OHE_DIM + PSSM_DIM  # 41 (21 + 20)
OUTPUT_DIM = len(SS_3_CLASSES) + 1 


# --- 2. Fonction d'Encodage ---

def one_hot_encode(sequence, char_to_int_map, dim, padding_char='<pad>', dtype=torch.float32):

    seq_len = len(sequence)
    one_hot = torch.zeros(seq_len, dim, dtype=dtype)
    
    for i, char in enumerate(sequence):
        idx = char_to_int_map.get(char, char_to_int_map[padding_char])
        one_hot[i, idx] = 1.0
            
    return one_hot


# --- 3. Définition du Jeu de Données PyTorch ---

class ProteinDataset(Dataset):
    def __init__(self, data_file_path, max_len=None):
        try:
            with open(data_file_path, 'r') as f:
                self.data = json.load(f)
        except FileNotFoundError:
            print(f"Erreur: Fichier de données non trouvé à l'emplacement: {data_file_path}")
            self.data = []
            return 
        
        if max_len is None:
            if not self.data:
                 self.max_len = 0
            else:
                 self.max_len = max([len(record['primary_sequence']) for record in self.data])
        else:
            self.max_len = max_len
        
        print(f"Dataset loaded. Max sequence length (L_max): {self.max_len}")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        record = self.data[idx]
        
        # 1. Préparation de la Séquence Primaire
        primary_seq = record['primary_sequence']
        L = len(primary_seq)
        
        # 2. Préparation du Label (Y) 
        raw_ss_seq = record['secondary_structure']
        ss_3_seq = ''.join([DSSP_3_MAP.get(ss, 'C') for ss in raw_ss_seq])
        Y_int = torch.tensor([SS_TO_INT.get(ss, SS_TO_INT['<pad>']) for ss in ss_3_seq], dtype=torch.long)

        # 3. One-Hot Encoding (OHE)
        X_ohe = one_hot_encode(primary_seq, AA_TO_INT, OHE_DIM) 
        
        # 4. Extraction et conversion du PSSM
        pssm_list = record.get('evolutionary')
        
        X_pssm = torch.tensor(pssm_list, dtype=torch.float32)

        if X_pssm.shape[0] == 20 and X_pssm.shape[1] == L:
    # Cas où le PSSM est (20 x L) mais doit être (L x 20)
            X_pssm = X_pssm.transpose(0, 1) # Transpose le tenseur en (L x 20)
        
        # 5. Concaténation des features (OHE [L x 21] + PSSM [L x 20]) -> [L x 41]
        X_combined = torch.cat([X_ohe, X_pssm], dim=1) 
        
        # 6. Padding (Remplissage)
        pad_size = self.max_len - L
        
        X_padded = torch.nn.functional.pad(X_combined.transpose(0, 1), (0, pad_size), 'constant', 0).transpose(0, 1)

        Y_padded = torch.nn.functional.pad(Y_int, (0, pad_size), 'constant', SS_TO_INT['<pad>'])

        return X_padded, Y_padded, L


# --- Exemple d'Utilisation ---

if __name__ == '__main__':
    file_path = 'testing_dataset.json' 

    test_dataset = ProteinDataset(file_path)
    
    if len(test_dataset) == 0:
        print("Le dataset est vide. Impossible de créer le DataLoader.")
    else:
        test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=True)
        
        X_batch, Y_batch, L_batch = next(iter(test_dataloader))
        
        print("\n--- Inspection des Tenseurs ---")
        print(f"Dimension Tenseur X (Input CNN): {X_batch.shape}")
        print(f"   => (Batch, L_max, D_input({INPUT_DIM}))")

        # Vérification des 21 premières dimensions (OHE) et des 20 suivantes (PSSM)
        print(f"Somme des dimensions OHE (doit être 1.0): {X_batch[0, 0, :OHE_DIM].sum()}")
        print(f"Valeurs PSSM (doit contenir des floats): {X_batch[0, 0, OHE_DIM:INPUT_DIM]}")

