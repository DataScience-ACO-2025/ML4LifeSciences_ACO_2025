# -*- coding: utf-8 -*-
"""
Created on Mon Nov 17 14:45:55 2025

@author: melin
"""

# -*- coding: utf-8 -*-
"""Classification d'images (pommes pourries/fraiches) sur Keras TensorFlow

Concernant le chargement des données du code, il va falloir créer un dossier pomme dans vos documents. 
Le dossier pomme doit contenir deux dossiers "mures" et "pourries" contenant les photos correspondantes.

Chargement des packages : nécessite l'installation préalable de tensorflow
"""

import matplotlib.pyplot as plt
import PIL
import time
import os
import tensorflow as tf

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential

"""On charge les données. """

# Définir le chemin local du dossier
path = "C:\\Users\\melin\\Documents\\pomme" # à modifier en fonction

# Vérification du nombre d'images
extensions = (".jpg", ".jpeg", ".png", ".bmp", ".gif")

count = sum(
    len([f for f in files if f.lower().endswith(extensions)])
    for _, _, files in os.walk(path)
)

print("Nombre total d'images :", count)

# Séparation des données en train et test
train_ds = tf.keras.utils.image_dataset_from_directory(
    path,
    validation_split=0.2,
    subset="training",
    seed=123,
    image_size=(128,128),
    batch_size=100,
    label_mode="binary",
)

val_ds = tf.keras.utils.image_dataset_from_directory(
    path,
    validation_split=0.2,
    subset="validation",
    seed=123,
    image_size=(128,128),
    batch_size=100,
    label_mode="binary"
)

"""Chargement des batchs à l'avance."""

train_ds = train_ds.prefetch(tf.data.AUTOTUNE)
val_ds   = val_ds.prefetch(tf.data.AUTOTUNE)

"""Création du modèle"""

data_augmentation = keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomZoom(0.2),
    layers.RandomRotation(0.1),
])


model = Sequential([
    keras.Input(shape=(128, 128, 3)),
    data_augmentation,
    layers.Rescaling(1./255),
    layers.Conv2D(16, 3, activation='relu'),
    layers.MaxPooling2D(),
    layers.Conv2D(32, 3, activation='relu'),
    layers.MaxPooling2D(),
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(1, activation='sigmoid')
])


"""Compilation du modèle"""

model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

"""Entrainement du modèle"""
start = time.time()

history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=10,
)


end = time.time()

print("Durée totale de l'entrainement :", (end-start)/60, "minutes")

"""Tracer les courbes d'accuracy et de loss"""

acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs_range = range(len(acc))

plt.figure(figsize=(10, 4))

plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label="Train Acc")
plt.plot(epochs_range, val_acc, label="Val Acc")
plt.title("Accuracy")
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label="Train Loss")
plt.plot(epochs_range, val_loss, label="Val Loss")
plt.title("Loss")
plt.legend()

plt.show()

