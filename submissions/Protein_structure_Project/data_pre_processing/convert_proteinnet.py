"""
Convert ProteinNet text files to JSON (valid JSON array).

exemple d'utilisation (copier-coller ça dans le terminal) : 

python data_pre_processing/convert_proteinnet.py casp8/training_30  training_30.json 

"""

import sys
import re
import json

NUM_DIMENSIONS = 3

_aa_dict = {'A': '0', 'C': '1', 'D': '2', 'E': '3', 'F': '4', 'G': '5', 'H': '6', 'I': '7',
            'K': '8', 'L': '9', 'M': '10', 'N': '11', 'P': '12', 'Q': '13', 'R': '14', 'S': '15',
            'T': '16', 'V': '17', 'W': '18', 'Y': '19'}
_dssp_dict = {'L': '0', 'H': '1', 'B': '2', 'E': '3', 'G': '4', 'I': '5', 'T': '6', 'S': '7'}

_mask_dict = {'-': '0', '+': '1'}

def letter_to_num(string, dict_):
    """Convert string of letters to list of ints."""
    patt = re.compile('[' + ''.join(dict_.keys()) + ']')
    num_string = patt.sub(lambda m: dict_[m.group(0)] + ' ', string)
    return [int(i) for i in num_string.split()]

def read_record(file_, num_evo_entries=20):
    """Read one ProteinNet record from a text file and convert to dict."""
    record = {}
    while True:
        line = file_.readline()
        if not line:
            return record if record else None
        line = line.strip()
        if line == '[ID]':
            record['id'] = file_.readline().strip()
        elif line == '[PRIMARY]':
            record['primary'] = letter_to_num(file_.readline().strip(), _aa_dict)
        elif line == '[EVOLUTIONARY]':
            evo = []
            for _ in range(num_evo_entries):
                evo.append([float(x) for x in file_.readline().split()])
            record['evolutionary'] = evo
        elif line == '[SECONDARY]':
            record['secondary'] = letter_to_num(file_.readline().strip(), _dssp_dict)
        elif line == '[TERTIARY]':
            tertiary = []
            for _ in range(NUM_DIMENSIONS):
                tertiary.append([float(x) for x in file_.readline().split()])
            record['tertiary'] = tertiary
        elif line == '[MASK]':
            record['mask'] = letter_to_num(file_.readline().strip(), _mask_dict)
        elif line == '':
            if record:
                return record

def protein_txt_to_json(input_path, output_path, num_evo_entries=20):
    """Convert a ProteinNet text file to JSON array."""
    records = []
    with open(input_path, 'r') as f:
        while True:
            record = read_record(f, num_evo_entries)
            if record is None:
                break
            records.append(record)

    with open(output_path, 'w') as out_file:
        json.dump(records, out_file, indent=2)
    print(f"Conversion done: {len(records)} records written to {output_path}")

if __name__ == '__main__':
    if len(sys.argv) < 3:
        print("Usage: python convert_proteinnet.py <input_file> <output_file> [num_evo_entries]")
        sys.exit(1)

    input_file = sys.argv[1]
    output_file = sys.argv[2]
    num_evo = int(sys.argv[3]) if len(sys.argv) > 3 else 20

    protein_txt_to_json(input_file, output_file, num_evo)
