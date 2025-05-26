import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# Función para cargar el modelo solo una vez y almacenarlo en caché
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("model/IAReciclaje.keras")

# Carga el modelo entrenado
model = load_model()
clases = ['Carton', 'Vidrio', 'Metal', 'Papel', 'Plastico']

# Título de la aplicación
st.title("Clasificador de Materiales Reciclables")

# Widget para subir una imagen
uploaded_file = st.file_uploader("Sube una imagen", type=["jpg", "jpeg", "png"])

# Función para preprocesar la imagen subida por el usuario
def preprocess_image(uploaded_file):
    img = Image.open(uploaded_file).convert("RGB")  # Abre y convierte la imagen a RGB
    img = img.resize((100, 100))                    # Redimensiona la imagen a 100x100 píxeles
    img_array = np.array(img).astype("float32") / 255.0  # Normaliza los valores de píxel
    img_array = np.expand_dims(img_array, axis=0)        # Añade una dimensión para el batch
    return img, img_array

# Si el usuario sube una imagen, se procesa y se muestra la predicción
if uploaded_file is not None:
    img, img_array = preprocess_image(uploaded_file)      # Preprocesa la imagen
    st.image(img, caption="Imagen subida", use_column_width=True)  # Muestra la imagen
    predict = model.predict(img_array)[0]                 # Realiza la predicción
    predicted_class = np.argmax(predict)                  # Obtiene la clase predicha
    st.write(f"**Tipo de Material:** {clases[predicted_class]}")   # Muestra el resultado