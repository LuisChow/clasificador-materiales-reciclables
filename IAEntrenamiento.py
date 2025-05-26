import tensorflow as tf
import numpy as np
import random
from PIL import Image
import cv2
from keras.callbacks import EarlyStopping
from sklearn.metrics import classification_report, confusion_matrix

# Descripciones y mapeo de clases
description = ("cardboard", "glass", "metal", "paper", "plastic")
maps = {"cardboard" : 0, "glass" : 1, "metal" : 2, "paper" : 3, "plastic" : 4}
    
numImgMax = 403

numTraining = round(numImgMax * 0.85)
numTesting = round(numImgMax * 0.15)

# Inicializa los arreglos para imágenes y etiquetas de entrenamiento y prueba
imgTraining = np.empty((numTraining * len(maps), 100,100, 3), dtype="uint8")
descTraining = np.empty(numTraining * len(maps), dtype="uint8")

imgTesting = np.empty((numTesting * len(maps), 100,100, 3), dtype="uint8")
descTesting = np.empty(numTesting * len(maps), dtype="uint8")

# Carga y preprocesa las imágenes de entrenamiento
for i in range(numTraining):
    for j in maps:
        image = Image.open(f"dataset/{j}/{j}{i + 1}.jpg")
        image = image.resize((100, 100))
        indice = i + maps[j] * numTraining
        imgTraining[indice] = np.array(image)
        descTraining[indice] = maps[j]

# Carga y preprocesa las imágenes de prueba
for i in range(numTesting):
    for j in maps:
        idx = i + numTraining
        image = Image.open(f"dataset/{j}/{j}{idx + 1}.jpg")
        image = image.resize((100, 100))
        indice = i + maps[j] * numTesting
        imgTesting[indice] = np.array(image)
        descTesting[indice] = maps[j]

# Normaliza los valores de las imágenes a [0, 1]
imgTraining = imgTraining.astype("float32") / 255.0
imgTesting = imgTesting.astype("float32") / 255.0

# Crea los datasets de TensorFlow para entrenamiento y prueba
train_dataset = tf.data.Dataset.from_tensor_slices((imgTraining, descTraining))
test_dataset = tf.data.Dataset.from_tensor_slices((imgTesting, descTesting))

# Baraja y agrupa los datos en lotes para entrenamiento y prueba
train_dataset = train_dataset.shuffle(numTraining * len(maps)).batch(32).prefetch(buffer_size=tf.data.AUTOTUNE)
test_dataset = test_dataset.batch(32).prefetch(buffer_size=tf.data.AUTOTUNE)

# Define la arquitectura de la red neuronal convolucional (CNN)
CNN = tf.keras.Sequential([
    tf.keras.layers.Conv2D(32, (5,5), activation='relu', input_shape=(100,100,3)),
    tf.keras.layers.MaxPooling2D((2,2)),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Conv2D(64, (5,5), activation='relu'),
    tf.keras.layers.MaxPooling2D((2,2)),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Conv2D(128, (5,5), activation='relu'),
    tf.keras.layers.MaxPooling2D((2,2)),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Conv2D(256, (5,5), activation='relu'),
    tf.keras.layers.MaxPooling2D((2,2)),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(480, activation='relu'),
    tf.keras.layers.Dense(5, activation='softmax')
])

# Compila el modelo con el optimizador, función de pérdida y métrica de precisión
CNN.compile(optimizer='adam', loss=tf.keras.losses.SparseCategoricalCrossentropy(), metrics=['accuracy'])

# Configura el callback de EarlyStopping para detener el entrenamiento si no mejora la validación
early_stop = EarlyStopping(
    monitor='val_loss',     
    patience=10,             
    restore_best_weights=True
)

# Entrena el modelo usando los datasets y el callback de EarlyStopping
CNN.fit(
    train_dataset,
    validation_data=test_dataset,
    epochs=150,
    callbacks=[early_stop] 
)

# Guarda el modelo entrenado en un archivo
CNN.save("model/IAReciclaje.keras")

# Evalúa el modelo: predice las clases del conjunto de prueba
y_true = descTesting
y_pred_probs = CNN.predict(imgTesting)
y_pred = np.argmax(y_pred_probs, axis=1)

# Imprime la matriz de confusión para analizar los aciertos y errores por clase
print("Matriz de confusión:")
print(confusion_matrix(y_true, y_pred))

# Imprime el reporte de clasificación con precisión, recall y f1-score por clase
print("Reporte de clasificación:")
print(classification_report(y_true, y_pred, target_names=description))