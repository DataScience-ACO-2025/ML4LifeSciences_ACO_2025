import json
import numpy as np
import pandas as pd
from aaindex import aaindex1


INPUT_JSON = "validation_matches_subset_dssp.json"  # chemin vers ton JSON
OUTPUT_XY = "XY_train.csv"   # chemin de sortie
WINDOW_SIZE = 17             # taille de la fenêtre
PADDING_AA = '-'             # caractère pour padding

SEQUENCE_AA = "ACDEFGHIKLMNPQRSTVWY-"
# Liste de descripteurs à utiliser
DESCRIPTORS = [
    'ARGP820101', 'BIGC670101', 'FAUJ880106','CHAM820101','GRAR740102',
    'RADA880108','FAUJ880111','FAUJ880112','BHAR880101','CHAM830107','FAUJ880109'
]

# Les 8 classes DSSP et leur mappage vers 3 classes
DSSP_8_CLASSES = 'HGEBITSL'
DSSP_3_MAP = {
    'H': 'H',  # Alpha Helix
    'G': 'H',  # 3-10 Helix
    'I': 'H',  # Pi Helix
    'E': 'E',  # Beta Strand
    'B': 'E',  # Beta Bridge
    'T': 'C',  # Turn
    'S': 'C',  # Bend
    'L': 'C',  # Loop / Coil
    '-': 'C',  # Missing → Coil
}

def encode_secondary_structure(ss_seq):
    """Mappe la structure secondaire DSSP (8 classes) vers 3 classes H/E/C"""
    return [DSSP_3_MAP.get(s, 'C') for s in ss_seq]



embedding_lookup = {}
for aa in SEQUENCE_AA:
    embedding_lookup[aa] = [aaindex1[d]['values'][aa] for d in DESCRIPTORS]

def load_json(json_path):
    """Charge un JSON et retourne la liste des entrées"""
    with open(json_path, 'r') as f:
        data = json.load(f)
    return data

def embed_aa(aa):
    """Retourne le vecteur embedding pour un résidu AA"""
    return embedding_lookup.get(aa, embedding_lookup['-'])

def sequence_to_windows(seq, window_size=17, pad_char='-'):
    """Découpe une séquence en fenêtres centrées, avec padding aux bords"""
    half_w = window_size // 2
    padded_seq = pad_char * half_w + seq + pad_char * half_w
    windows = [padded_seq[i - half_w: i + half_w + 1] for i in range(half_w, len(seq) + half_w)]
    return windows

def window_to_flat_embedding(window):
    """Transforme une fenêtre de résidus en vecteur flatten"""
    flat_vector = []
    for aa in window:
        flat_vector.extend(embed_aa(aa))
    return flat_vector


def prepare_tabular_dataset(INPUT_JSON, window_size=17, save_as_CSV=None):
    """
    Transforme un dataset JSON en X et y tabulaire pour Random Forest.
    
    INPUT_JSON : chemin vers le JSON contenant les séquences
    window_size : taille de la fenêtre (doit être impair)
    save_as_CSV : chemin pour sauvegarder le CSV, None pour ne pas sauvegarder
    """
    data = load_json(INPUT_JSON)
    X_list, y_list = [], []

    for entry in data:
        seq = entry['primary_sequence']
        ss = entry['secondary_structure']
        ss_encoded = encode_secondary_structure(ss) 
        windows = sequence_to_windows(seq, window_size)
        X_list.extend(window_to_flat_embedding(w) for w in windows)
        y_list.extend(ss_encoded)

    X = np.array(X_list)
    y = np.array(y_list)

    if save_as_CSV:
        df = pd.DataFrame(X)
        df['target'] = y
        df.to_csv(save_as_CSV, index=False)

    return X, y


if __name__ == "__main__":
    
    # Préparer X et y
    X, y = prepare_tabular_dataset(INPUT_JSON, window_size=WINDOW_SIZE)
    
    # Créer un DataFrame pour sauvegarde
    df_X = pd.DataFrame(X)
    df_X['target'] = y
    df_X.to_csv(OUTPUT_XY, index=False)
    
    print(f"✅ Dataset généré : {OUTPUT_XY} | {X.shape[0]} lignes, {X.shape[1]} features")