import os
import csv

# Dossier contenant les fichiers audio
DOSSIER = "C:/Users/augus/OneDrive/Documents/Cours/ACO/M2/Machine Learning/Projet chat/DeepMeows/dataset"
# Nom du fichier CSV de sortie
CSV_SORTIE = "metadata.csv"

def extraire_infos(nom_fichier):
    """
    Prend un nom de fichier comme 'B_ANI01_MC_FN_SIM01_101'
    et renvoie une liste des différentes parties.
    """
    # Retirer l’extension si elle existe
    nom_sans_ext = os.path.splitext(nom_fichier)[0]
    # Découper selon les underscores
    parties = nom_sans_ext.split("_")
    return parties

def main():
    # Récupérer la liste de tous les fichiers du dossier
    fichiers = [f for f in os.listdir(DOSSIER) if os.path.isfile(os.path.join(DOSSIER, f))]
    
    # Créer une liste pour stocker les données
    lignes_csv = []

    for fichier in fichiers:
        parties = extraire_infos(fichier)
        ligne = parties + [fichier]  # Ajouter le nom complet du fichier à la fin
        lignes_csv.append(ligne)

    # Déterminer le nombre maximal de parties
    max_colonnes = max(len(ligne) for ligne in lignes_csv)

    # Créer les noms de colonnes
    noms_colonnes = ["emission_context","cat_ID","breed","sex","owner_ID","session","file_name"]

    # Écrire le CSV
    with open("C:/Users/augus/OneDrive/Documents/Cours/ACO/M2/Machine Learning/Projet chat/DeepMeows/metadata.csv", mode="w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(noms_colonnes)
        writer.writerows(lignes_csv)

    print(f"Fichier CSV créé : {CSV_SORTIE}")
    print(f"Nombre de fichiers traités : {len(lignes_csv)}")

if __name__ == "__main__":
    main()
