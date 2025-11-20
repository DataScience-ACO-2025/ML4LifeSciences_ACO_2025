---
title: "Evaluation par les pairs"
output:
  pdf_document: default
  word_document: default
---

### Grille d’évaluation croisée

:::info
À remplir par chaque groupe pour chaque autre groupe, sur une échelle de 1 à 5, avec 5 = excellent). A remplir pour le **17 novembre** impérativement.
:::

Groupe évaluateur : 1 

Groupe évalué : 3

   **Critère**                          | **Sous-critères**                                                                 | **Note (1-5)** | **Commentaires**                     |
 |--------------------------------------|-----------------------------------------------------------------------------------|----------------|--------------------------------------|
 | **1. Pertinence du thème**           | Le thème choisi est-il original et complémentaire aux enseignements existants ?       |      5          |        Sujet très pertinent : l’utilisation d’un Vision Transformer (ViT) pour détecter maladies et espèces végétales à partir de PlantDoc est originale dans le cadre du cours. Le choix d’un modèle moderne (ViT) + le domaine de l’agronomie/santé des plantes rend le thème à la fois original.                              |
 |           |  L’ancrage en sciences du vivant est-il justifié et bien illustré ?      |     4           |          L’ancrage est présent (données PlantDoc, objectifs agronomiques) et bien illustré dans les slides. Mais l’analyse biologique reste limitée : peu d’explications sur les maladies elles-mêmes, leur impact.                          |
 | **2. Structure et clarté**           | La présentation suit-elle une logique pédagogique (méthodologie → technique → application) ?      |        5        |           La progression est logique et schémas explicites                           |
  |           | Le niveau de détail permet-il une bonne compréhension du sujet ?      |      4          |                              Les explications sur ViT sont synthétiques et pédagogiques. La seule limite manque d'infos sur la gestion des classes, split des données et justification des hyperparamètres.        |
 | **3. Qualité méthodologique**        | Les concepts clés sont-ils expliqués de manière accessible mais rigoureuse ?        |        4        |              La structure du ViT est bien expliquée. Les résultats sont bien interprétés.                        |
  |        | Les choix méthodologiques sont-ils argumentés ?         |        4        |     Ils expliquent clairement :pourquoi ViT est choisi, raisons de résultats faibles. Juste besoin de plus de justification sur 10 epochs seulement, absence de data augmentation...
                               |
 | **4. Rigueur technique**             | Le code (GitHub) est-il fonctionnel et commenté  |                                   4                   | Le code fonctionne (importation, modèle, training loop, confusion matrix) mais pas d’explication dans le README du Git.
  |             | Les résultats sont-ils reproductibles (par ex. y a t-il des graines pour générer les échantillons ?) |         3.5       |            modèles bien définis, pas de seed dans le code                          |
           |  | Les données sont-elles bien documentées et adaptées à l’application ?  |         5                                         | PlantDoc est un dataset cohérent pour ce type d’étude. Les slides montrent graphiques de répartition et classes.
   |            | Y a-t-il des erreurs techniques ou des approximations ?  |      4          |                                      |
| **5. Ancrage applicatif**            | L’application en sciences du vivant est-elle pertinente et bien illustrée ?  |         5       |                   L’application est claire détection de maladies foliaires, enjeux agronomiques, amélioration du diagnostic.                   |
|        | Les limites de la méthode/thème sont-elles mentionnées ??  |       5         |       Très bonne discussion.                               |
 | **6. Qualité des supports**          | Les slides sont-ils synthétiques et d'une forme adaptée ? |  5 |
  |     Slides très soignés.     | Le dépôt GitHub est-il organisé (README, structure claire, données accessibles) ? | 3 | Code présent mais manque de README,explication des scripts.
   |           | Les documents partagés en amont (code, données, slides) sont-ils utilisables et complets pour une réutilisation ? | 4 |
|          | Les sources sont-elles citées et fiables ? | 5