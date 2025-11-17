import os
import random
import shutil
import sys
import traceback

def compter_images_par_sousdossier(dossier):
    extensions_images = {".jpg", ".jpeg", ".png", ".gif", ".bmp", ".tiff"}
    resultats = {}

    for racine, sous_dossiers, fichiers in os.walk(dossier):
        if racine == dossier:
            continue

        nb_images = sum(1 for f in fichiers if os.path.splitext(f)[1].lower() in extensions_images)
        nom_sous_dossier = os.path.basename(racine)
        resultats[nom_sous_dossier] = nb_images

    return resultats

def generer_groupes(dossier_principal):
    try:
        # Normaliser le chemin et vérifier
        dossier_principal = os.path.abspath(dossier_principal)
        print(f"[INFO] dossier_principal résolu en : {dossier_principal}")

        if not os.path.exists(dossier_principal):
            raise FileNotFoundError(f"Le dossier principal n'existe pas : {dossier_principal}")

        if not os.path.isdir(dossier_principal):
            raise NotADirectoryError(f"Le chemin n'est pas un dossier : {dossier_principal}")

        extensions_images = {".jpg", ".jpeg", ".png", ".gif", ".bmp", ".tiff"}
        tailles = [5, 10, 50, 100, 500]

        # Lister uniquement les sous-dossiers directs (exclut fichiers)
        sous_dossiers = [
            os.path.join(dossier_principal, d)
            for d in os.listdir(dossier_principal)
            if os.path.isdir(os.path.join(dossier_principal, d))
        ]

        print(f"[INFO] Sous-dossiers trouvés ({len(sous_dossiers)}): {[os.path.basename(s) for s in sous_dossiers]}")

        # ----------------------------------------------------------
        # 1) CRÉER LE JEU DE DONNÉES INDÉPENDANT : 100 IMAGES/ESPÈCE
        # ----------------------------------------------------------
        dossier_indep = os.path.join(dossier_principal, "100_img_indep")
        try:
            os.makedirs(dossier_indep, exist_ok=True)
            print(f"[OK] Dossier indépendant créé (ou existait déjà) : {dossier_indep}")
        except Exception as e:
            raise OSError(f"Impossible de créer le dossier indépendant '{dossier_indep}': {e}")

        images_indep_par_espece = {}  # Pour mémoriser les images à exclure

        print("\n=== Création du dataset indépendant (100_img_indep) ===")

        for sous in sous_dossiers:
            # Skip the output folders if script was re-run
            if os.path.basename(sous).endswith("_img") or os.path.basename(sous) == "100_img_indep":
                print(f"[SKIP] Ignoré (dossier de sortie ou dataset indépendant) : {os.path.basename(sous)}")
                continue

            images = [
                os.path.join(sous, f)
                for f in os.listdir(sous)
                if os.path.splitext(f)[1].lower() in extensions_images
            ]

            if len(images) < 100:
                print(f"[INFO] {os.path.basename(sous)} ignoré pour dataset indépendant : moins de 100 images ({len(images)})")
                continue

            nom_sous = os.path.basename(sous)
            dossier_espece_indep = os.path.join(dossier_indep, nom_sous)
            os.makedirs(dossier_espece_indep, exist_ok=True)

            # Tirage aléatoire des 100 images pour le dataset indépendant
            images_indep = random.sample(images, 100)
            images_indep_par_espece[nom_sous] = set(images_indep)

            # Copie des images
            for img in images_indep:
                shutil.copy(img, dossier_espece_indep)

            print(f"  → Dataset indépendant : {nom_sous} = 100 images OK")

        # --------------------------------------------------------------------
        # 2) CRÉATION DES AUTRES JEUX (5,10,50,100,500) EN EXCLUANT LES 100 OK
        # --------------------------------------------------------------------
        print("\n=== Création des datasets multiples (5,10,50,100,500) ===")

        for sous in sous_dossiers:
            nom_sous = os.path.basename(sous)

            if nom_sous == "100_img_indep" or nom_sous.endswith("_img"):
                # On ignore les dossiers de sortie s'ils existent dans le dossier principal
                continue

            images = [
                os.path.join(sous, f)
                for f in os.listdir(sous)
                if os.path.splitext(f)[1].lower() in extensions_images
            ]

            # Retirer les images du dataset indépendant
            if nom_sous in images_indep_par_espece:
                images = [img for img in images if img not in images_indep_par_espece[nom_sous]]

            # Vérifie qu'il reste au moins 500 images
            if len(images) < 500:
                print(f"[INFO] {nom_sous} : ignoré (pas assez d'images après exclusion) => {len(images)} restantes")
                continue

            print(f"[OK] {nom_sous} : {len(images)} images restantes → création des sous-ensembles")

            # Création des datasets pour chaque taille
            for t in tailles:
                dossier_taille = os.path.join(dossier_principal, f"{t}_img")
                os.makedirs(dossier_taille, exist_ok=True)

                dossier_cible = os.path.join(dossier_taille, nom_sous)
                os.makedirs(dossier_cible, exist_ok=True)

                images_sel = random.sample(images, t)

                for img in images_sel:
                    shutil.copy(img, dossier_cible)

                print(f"  → {t}_img/{nom_sous} rempli avec {t} images")

        print("\n[TASK DONE] Traitement terminé.")

    except Exception as e:
        print("\n[ERROR] Une exception est survenue :")
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    # <-- Modifie ce chemin si nécessaire (chemin relatif OK mais on le résoud en absolu)
    dossier_cible = "../Insect_Detect_classification_v2"
    generer_groupes(dossier_cible)
