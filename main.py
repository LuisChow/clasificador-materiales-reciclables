import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# Tamano de entrada del nuevo modelo (MobileNetV2 trabaja con 224x224)
IMG_SIZE = (224, 224)


# Funcion para cargar el modelo solo una vez y almacenarlo en cache
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("model/IAReciclaje.keras")


# Carga el modelo entrenado con transfer learning
model = load_model()
clases = ['Carton', 'Vidrio', 'Metal', 'Papel', 'Plastico']

# Titulo de la aplicacion
st.title("Clasificador de Materiales Reciclables")
st.caption(
    "Modelo basado en transfer learning (MobileNetV2). "
    "Sube una foto de un material y la IA predice su categoria."
)

# Widget para subir una imagen
uploaded_file = st.file_uploader("Sube una imagen", type=["jpg", "jpeg", "png"])


# Funcion para preprocesar la imagen subida por el usuario
def preprocess_image(uploaded_file):
    img = Image.open(uploaded_file).convert("RGB")   # Abre y convierte la imagen a RGB
    img_resized = img.resize(IMG_SIZE)               # Redimensiona a 224x224 pixeles
    # El modelo incluye su propia capa de normalizacion (Rescaling),
    # asi que se entregan los pixeles en su rango original [0, 255].
    img_array = np.array(img_resized).astype("float32")
    img_array = np.expand_dims(img_array, axis=0)    # Anade la dimension del batch
    return img, img_array


# Si el usuario sube una imagen, se procesa y se muestra la prediccion
if uploaded_file is not None:
    img, img_array = preprocess_image(uploaded_file)
    st.image(img, caption="Imagen subida", use_container_width=True)

    predict = model.predict(img_array)[0]            # Realiza la prediccion
    predicted_class = int(np.argmax(predict))        # Obtiene la clase predicha
    confidence = float(predict[predicted_class]) * 100

    st.write(f"**Tipo de Material:** {clases[predicted_class]}")
    st.write(f"**Confianza:** {confidence:.1f}%")

    # Desglose de la probabilidad asignada a cada categoria
    st.subheader("Probabilidad por categoria")
    for nombre, prob in sorted(zip(clases, predict), key=lambda par: par[1], reverse=True):
        st.write(f"{nombre}: {prob * 100:.1f}%")
        st.progress(float(prob))
