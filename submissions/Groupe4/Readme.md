Notre projet s’inscrit dans le cadre de l’apprentissage profond appliqué à la classification d’images. L’objectif est de développer, entraîner et évaluer un réseau de neurones convolutionnel (CNN) capable de reconnaître les espèces présentes dans le jeu de données Animals10, composé d’images réparties en 10 classes d’animaux.

# Objectif du projet

L’objectif principal est de construire un pipeline complet de deep learning comprenant :
le prétraitement des images, la création et le réglage de l’architecture du modèle, l’entraînement et la validation du CNN, l’analyse des performances et des erreurs, et l’exploration de techniques pour améliorer la robustesse du modèle, notamment via la data augmentation.

# Référence : code du cours de Laetitia Chapel

Pour assurer une base solide et reproductible, nous nous sommes appuyés sur le code fourni dans le cours de Laetitia Chapel (module Deep Learning).
Ce code nous a servi de référence structurante, notamment pour :
l’organisation du pipeline PyTorch, la construction de l’architecture CNN, la préparation des dataloaders, la mise en place des fonctions d’entraînement et de test.
À partir de cette base, nous avons adapté, modifié et enrichi le code pour l’appliquer à notre dataset Animals10 et intégrer nos propres expérimentations (data augmentation, variations du CNN, analyse des performances…).

# Jeu de données : Animals10

Le dataset contient environ 28 000 images, réparties en 10 catégories.
Les images sont très hétérogènes (qualité, orientation, luminosité), ce qui rend la classification réaliste et non triviale.

# Modèle choisi : CNN simple (PyTorch)

Nous avons conçu un CNN léger et pédagogique, inspiré du modèle utilisé en cours.
L’architecture comporte :
- des couches convolutionnelles,
- des activations ReLU,
- du max-pooling,
- des couches fully connected.
Ce choix permet de bien comprendre l’impact de chaque bloc sur l’apprentissage, tout en obtenant des performances satisfaisantes.

# Organisation de notre Github
models
→ cnn_Color.ipynb
→ cnn_Sup_Zone.ipynb
→ test_CNN.py

data
→ dataset Animals10

scripts
→ Animals.py : chargement + preprocessing

presentation
→ slides du projet

📄 README.md
→ synthèse du projet
