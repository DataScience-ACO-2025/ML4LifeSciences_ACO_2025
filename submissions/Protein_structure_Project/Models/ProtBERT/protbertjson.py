# protbert_embeddings.py
# Objectif : Générer et sauvegarder progressivement les embeddings ProtBERT

from transformers import AutoTokenizer, AutoModel
import torch
import numpy as np
import json
import os
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore")

# Charger le JSON

def load_json(json_path):
    with open(json_path, 'r') as f:
        data = json.load(f)
    return data

json_path = "testing_dataset"  # JSON déjà mappé avec DSSP
dataset = load_json(json_path)
print(f"{len(dataset)} séquences chargées depuis {json_path}")


# Charger ProtBERT

print("Chargement de ProtBERT...")
tokenizer = AutoTokenizer.from_pretrained("Rostlab/prot_bert", do_lower_case=False)
model = AutoModel.from_pretrained("Rostlab/prot_bert")
model.eval()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
print(f"Appareil utilisé : {device}")

def embed_sequence(seq):
    seq = " ".join(list(seq))
    tokens = tokenizer(seq, return_tensors="pt").to(device)
    with torch.no_grad():
        output = model(**tokens)
    emb = output.last_hidden_state.squeeze(0)[1:-1].cpu().numpy()
    return emb

# Génération et sauvegarde progressive

output_dir = "embeddings_testing"
os.makedirs(output_dir, exist_ok=True)
labels_list = []

for i, sample in enumerate(tqdm(dataset, desc="Génération des embeddings", dynamic_ncols=True)):
    seq = sample["primary_sequence"]
    emb = embed_sequence(seq)
    np.save(os.path.join(output_dir, f"seq_{i}.npy"), emb)

    # Convertir H/E/C en indices 0/1/2
    ss = sample["secondary_structure"]
    label_idx = np.array([{"H":0, "E":1, "C":2}.get(c, 2) for c in ss])
    labels_list.append(label_idx)

labels_array = np.array(labels_list, dtype=object)
np.save(os.path.join(output_dir, "labels.npy"), labels_array)

print(f"Embeddings et labels sauvegardés dans {output_dir}")
