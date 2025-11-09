# MachineLearning4LifeSciences_ACO_2025

# Suggestion structure du repo 

```text
Suggestion structure du repo
/ (racine)
│
├── README.md
│
├── docs/ ← dossier de sortie HTML publié (site web commpilé) 
│
├── /website/ ← site Quarto pour hébergement sur GitHub Pages
│   ├── _quarto.yml ← configuration du site
│   ├── index.qmd ← page d’accueil (programme, encadrants, calendrier)
│   └── /projects/ ← pages web par mini-cours (une par groupe)
│   
│
├── /submissions/ ← contributions des groupes
│   ├── groupe01_nom_projet/
│   │   ├── README.md ← résumé du projet
│   │   ├── slides.pdf ← diaporama du mini-cours
│   │   ├── notebook.ipynb ← code exécutable
│   │   ├── data/ ← données utilisées (ou lien vers source)
│   │   └── requirements.txt ← dépendances
│   ├── groupe02_nom_projet/
│   └── ...
│
└── /reviews/ ← évaluations croisées entre groupes
    ├── groupe01_review_groupe02.md
    └── ...


