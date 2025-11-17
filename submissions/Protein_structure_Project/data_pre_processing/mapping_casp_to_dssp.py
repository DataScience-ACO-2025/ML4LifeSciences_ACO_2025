"""
    Compare un subset de protéines avec le dataset DSSP sur la séquence primaire.
    Retourne un json avec les séquences, les labels DSSP correspondants etc ... 
    
    exemple d'utilisation (à copier coller dans le terminal) : 
    python mapping_casp_to_dssp("casp8/validation.json", "full_protein_dssp_annotations.json", "validation_dataset.json")
"""
import sys
import json

aa_alphabet = list('ACDEFGHIKLMNPQRSTVWY')

def mapping_casp_to_dssp(casp_json_path, dssp_json_path, save_path=None):
    """
    casp_json_path : le fichier casp au format json (train ou valid ou test )
    dssp_json_path : le fichier avec les labels des structures secondaires
    save_path : le nom de l'output 
    """

    with open(casp_json_path, 'r') as f:
        subset_data = json.load(f)  # liste de dicts avec 'id' et 'primary'

    with open(dssp_json_path, 'r') as f:
        dssp_data = json.load(f)  
    
    # Création d'un lookup par séquence primaire dans DSSP
    seq_to_dssp_entry = {}
    for dssp_id, entry in dssp_data.items():
        seq = entry['Sequence']
        seq_to_dssp_entry[seq] = {
            'dssp_id': dssp_id,
            'DSSP': entry.get('DSSP', '')
        }

    matches_info = []
    for record in subset_data:
        seq_train = ''.join([aa_alphabet[idx] for idx in record['primary']])
        if seq_train in seq_to_dssp_entry:
            dssp_entry = seq_to_dssp_entry[seq_train]
            matches_info.append({
                'subset_id': record['id'],
                'dssp_id': dssp_entry['dssp_id'],
                'primary_sequence': seq_train,
                'secondary_structure': dssp_entry['DSSP'],
                'mask': record.get('mask', None),
                'evolutionary': record.get('evolutionary', None),
                'tertiary': record.get('tertiary', None)
            })

    if save_path:
        with open(save_path, 'w') as out_file:
            json.dump(matches_info, out_file, indent=2)

    print(f"Total sequences in subset: {len(subset_data)}")
    print(f"Matches found in DSSP: {len(matches_info)}")
    print(f"Sequences not found in DSSP: {len(subset_data) - len(matches_info)}")

    


if __name__ == '__main__':
    if len(sys.argv) < 3:
        print("Usage: python mapping_casp_to_dssp.py <casp_json_path> <dssp_json_path> <save_path>")
        sys.exit(1)

    casp_json_path = sys.argv[1]
    dssp_json_path = sys.argv[2]
    save_path = sys.argv[3]


    mapping_casp_to_dssp(casp_json_path, dssp_json_path, save_path)