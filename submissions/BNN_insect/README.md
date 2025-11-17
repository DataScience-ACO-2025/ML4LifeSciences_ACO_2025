# BNN_insect

## Description Projet

  Notre projet s'axe sur des problématiques d'apprentissage profond via réseaux de neuronnes, dans le cadre de la reconnaissance d'images. Il a pour but l'évaluation de l'apport d'un modèle comportant des composantes bayésiennes par rapport à un modèle purement fréquentiste. En comparant les 2 modèles du point de vue de l'acuracy en fonction des ressources demandées (nombre d'image dans le jeu de donnée, temps d'entrainement du modèle, ressource GPU, ...) nous pourrons alors déterminer dans quels cas il est préférable d'utiliser l'une des 2 méthodes. 
  
  En terme de donnée, nous avons trouvé un jeu de donnée "Insect_Detect_classification_v2" contenant des images d'insectes de basse qualité (70x70) répartie dans 27 classes. 
  
  Pour effectuer cette démarche, le choix du modèle de CNN s'est d'abord imposé. Notre réflexion s'est portée autour 3 modèles connus et éprouvés : AlexNet, VGGNet et ResNet. Ces 3 modèles ont été entrainé avec les data d'"ImageNet", banque de plus de 14 millions d'images annotées à la main(1.2M à l'époque). La simplicité de manipulation du modèle ainsi que sa clarté ont été les principaux éléments de décisions qui nous a poussé à utiliser AlexNet, qui fait figure de basique du genre. En effet, ce modèle contient 8 couches : 5 convolutionnelles (+ 3 max-pooling) et 3 couches denses.
  
  Nous possédons des ressources limitées, et avons donc fait le choix de prendre le modèle AlexNet déjà pré-entrainé (en conservant les poids optenu à la suite de son entrainement sur ImageNet) et de le fine-tuner pour nos données. 
  
  Ainsi, reste le problème du modèle bayésien. Au cours de nos recherches, nous avons découvert un package permettant de fine-tuner AlexNet en remplacant les couches convolutionnelles et linéaires classiques, qui renvoient des poids w fixe, en couches bayésiennes qui renvoient une distriution de poids w. 
  
  L'objectif est alors de comprendre ce package, réussir à le faire fonctionner correctement pour notre problématique, et l'optimiser au possible avec nos ressources disponibles.
  
## Plan du Git

 Le code du CNN : .models/bon_CNN

 Le code du BNN : TEST_BNN

 Le code qui explique les difficultées avec le BNN : 

 La donnée : .data

 La littérature scientifique : .literature

 Les slides : .presentation
