# =======================================================================================
#                     Installation : à faire uniquement la première fois 
# =======================================================================================
  

#############  1. Installation des packages ##############
# (uniquement si pas déjà installés)
install.packages("reticulate")
install.packages("tensorflow")
install.packages("tictoc")                  # optionnel, calculs des temps de calcul 
install.packages("keras3")
install.packages("keras")


############# 2. Chargement des packages   ##############
library(keras3)
library(tensorflow)
library(dplyr)
library(keras)
library(tictoc)
library(reticulate)


#############    3. Installation de python   ############## 
# si pas déjà installé en local sur la machine
reticulate::py_available()
reticulate::install_python(version = "3.9") # version minimum de python (si besoin)


#############  4. Installation de keras    ##############

############# Option 1 pour installer keras #############
keras::install_keras()

############# Option 2 Pour installer keras ############# 
# si la première option ne fonctionne pas
keras::install_keras(
  # Méthode : environnement virtuel
  method = "virtualenv",
  # Nom de l'environnement à créer
  envname = "r-keras",
  # Version de Python connue pour sa stabilité
  python_version = "3.9", 
  # Version de TensorFlow recommandée
  tensorflow ="2.10.0"
)


############# 5. Chargement des packages   ##############

tensorflow::tf_config()  # Check TensorFlow version
keras::is_keras_available()


# =======================================================================================
#                    Si déjà tout est installé : Chargement des packages  
# =======================================================================================

library(keras3)
library(tensorflow)
library(dplyr)
library(keras)
library(tictoc)
library(reticulate)



# =======================================================================================
#                    Préparation des fichiers et définition des paramètres 
# =======================================================================================

# PREREQUIS
# Dans le même environnement que ce workflow, il faut télécharger les données       => possibilité de changer le nom du dossier 
# Un dossier (ici appelé 'Pomme') contient deux sous dossiers 'mures' & 'pourries'  => possibilité de changer le nom des sous dossiers (penser à changer les éléments du vecteur "classes")
# Dans 'pourries' => photos des 'rotten apples' 
# Dans 'mures' => photos des 'fresh apples' 

base_dir <- "C:\\Documents\\Agrocampus\\M2\\DL\\Conference\\Machine-Learning-Conference\\Pomme"  # remplacer par le path vers les données 
classes <- c("mures", "pourries")
split_ratio <- 0.8
set.seed(42)

# création du path vers les nouveaux dossiers 'train' et 'test' 
train_dir <- file.path(base_dir, "train")  
test_dir <- file.path(base_dir, "test")

dir.create(train_dir, recursive = TRUE, showWarnings = FALSE)
dir.create(test_dir, recursive = TRUE, showWarnings = FALSE)



# =======================================================================================
#                                 Déplacement des images  
# =======================================================================================

# On va déplacer chaque image dans train ou test avec le split ratio défini plus haut 
# dans le dossier "train" seront déplacées les 80 % des images de fresh apples, et 80 % des images de roten apple

tic("Préparation des répertoires et partage des fichiers")

for (class in classes) {
  cat("Traitement de la classe:", class, "\n")
  
  source_path <- file.path(base_dir, class)
  train_class_path <- file.path(train_dir, class)
  test_class_path <- file.path(test_dir, class)
  
  dir.create(train_class_path, showWarnings = FALSE)
  dir.create(test_class_path, showWarnings = FALSE)
  
  all_files <- list.files(source_path, full.names = TRUE)
  num_files <- length(all_files)
  
  if (num_files == 0) {
    cat("  !!! Aucun fichier trouvé dans", source_path, "!!!\n")
    next
  }
  
  train_indices <- sample(seq_len(num_files), size = floor(split_ratio * num_files))
  train_files <- all_files[train_indices]
  test_files <- all_files[-train_indices]
  
  file.copy(from = train_files, to = file.path(train_class_path, basename(train_files)))
  file.copy(from = test_files, to = file.path(test_class_path, basename(test_files)))
  
  cat("  ->", length(train_files), "fichiers déplacés vers 'train/", class, "'\n", sep="")
  cat("  ->", length(test_files), "fichiers déplacés vers 'test/", class, "'\n", sep="")
}

toc()  # Fin de la préparation des données



# =======================================================================================
#                                Prétraitement des images 
# =======================================================================================

tic("Prétraitement des images")

image_size <- c(128, 128) # 128 par 128 pixels : format des images 

train_datagen <- image_data_generator(
  rescale = 1/255,
  shear_range = 0.2,
  zoom_range = 0.2,
  horizontal_flip = TRUE
)

test_datagen <- image_data_generator(
  rescale = 1/255
)

train_generator <- flow_images_from_directory(
  train_dir,
  generator = train_datagen,
  target_size = image_size,
  batch_size = 100,
  class_mode = 'binary'
)

test_generator <- flow_images_from_directory(
  test_dir,
  generator = test_datagen,
  target_size = image_size,
  batch_size = 100,
  class_mode = 'binary'
)

toc()  # Fin du prétraitement des images




# =======================================================================================
#                        Construction et compilation du modèle 
# =======================================================================================

tic("Construction et compilation du modèle")

model <- keras_model_sequential() %>%
  layer_conv_2d(filters = 16, kernel_size = c(3, 3), activation = 'relu') %>%
  layer_max_pooling_2d(pool_size = c(2, 2)) %>%
  layer_conv_2d(filters = 32, kernel_size = c(3, 3), activation = 'relu') %>%
  layer_max_pooling_2d(pool_size = c(2, 2)) %>%
  layer_flatten() %>%
  layer_dense(units = 64, activation = 'relu') %>%
  layer_dense(units = 1, activation = 'sigmoid')

model %>% compile(
  optimizer = optimizer_adam(),
  loss = 'binary_crossentropy',
  metrics = c('accuracy')
)

# Affichage du résumé du modèle
summary(model)

toc()  # Fin de la construction du modèle



# =======================================================================================
#                                  Entraînement du modèle
# =======================================================================================

tic("Entrainement du modèle")

options(keras.view_metrics = FALSE)

EPOCHS <- 10

history <- model %>% fit(
  train_generator,
  steps_per_epoch = ceiling(train_generator$n / train_generator$batch_size),
  epochs = EPOCHS,
  validation_data = test_generator,
  validation_steps = ceiling(test_generator$n / test_generator$batch_size),
  view_metrics = FALSE  
)

toc()  # Fin de l'entraînement



# =======================================================================================
#                                  Evaluation du modèle
# =======================================================================================

tic("Évaluation du modèle")

evaluation <- model %>% evaluate(
  test_generator,
  steps = ceiling(test_generator$n / test_generator$batch_size)
)

cat('Test loss:', evaluation$loss, '\n')
cat('Test accuracy:', evaluation$accuracy, '\n')

toc()  # Fin de l'évaluation



# =======================================================================================
#                         Affichage des courbes de performances
# =======================================================================================

# Plot loss
plot(history$metrics$loss, type='l', col='blue', xlab='Epoch', ylab='Loss',
     main='Loss', lwd =2)
lines(history$metrics$val_loss, col='orange', lwd =2)
legend('topright', legend=c('Training Loss','Validation Loss'), col=c('blue','orange'), lty=1)

# Plot accuracy
plot(history$metrics$accuracy, type='l', col='blue', xlab='Epoch',
     main='Accuracy', lwd =2)
lines(history$metrics$val_accuracy, col='orange', lwd = 2)
legend('bottomright', legend=c('Training Accuracy','Validation Accuracy'), col=c('blue','orange'), lty=1)



