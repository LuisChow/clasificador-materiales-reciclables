# Clasificador de Materiales Reciclables
> **Demo en vivo:** https://clasificador-materiales-reciclables-gw5qby2fdrqyngha4smndc.streamlit.app/

Aplicación de visión por computadora que clasifica imágenes de residuos en cinco categorías de materiales reciclables —cartón, vidrio, metal, papel y plástico— mediante una red neuronal convolucional (CNN) entrenada con TensorFlow. Incluye una interfaz web en Streamlit donde el usuario sube una imagen y recibe la predicción al instante.

## Características

- **Clasificación en 5 categorías:** cartón, vidrio, metal, papel y plástico.
- **Red neuronal convolucional** entrenada desde cero con TensorFlow / Keras.
- **Interfaz web interactiva** en Streamlit: subir imagen y ver el resultado en tiempo real.
- **Modelo preentrenado incluido** (`model/IAReciclaje.keras`) — se puede usar la aplicación sin necesidad de reentrenar.
- **Evaluación del modelo** con matriz de confusión y reporte de clasificación (precisión, recall y F1 por clase).

## Stack tecnológico

- **Python**
- **TensorFlow / Keras** — definición y entrenamiento de la CNN
- **Streamlit** — interfaz web
- **NumPy** — manejo de arreglos e imágenes
- **Pillow (PIL)** — carga y preprocesamiento de imágenes
- **scikit-learn** — métricas de evaluación (matriz de confusión, classification report)

## Arquitectura del modelo

La red recibe imágenes de 100×100 píxeles en RGB y está compuesta por cuatro bloques convolucionales seguidos de una etapa densa de clasificación:

| Capa | Configuración |
|------|---------------|
| Conv2D + MaxPooling + Dropout | 32 filtros (5×5), dropout 0.3 |
| Conv2D + MaxPooling + Dropout | 64 filtros (5×5), dropout 0.3 |
| Conv2D + MaxPooling + Dropout | 128 filtros (5×5), dropout 0.3 |
| Conv2D + MaxPooling + Dropout | 256 filtros (5×5), dropout 0.5 |
| Flatten + Dense | 480 neuronas (ReLU) |
| Dense (salida) | 5 neuronas (Softmax) |

- **Optimizador:** Adam
- **Función de pérdida:** Sparse Categorical Crossentropy
- **Entrenamiento:** hasta 150 épocas con *EarlyStopping* (paciencia de 10 épocas, restaurando los mejores pesos)

## Dataset

El conjunto de datos contiene imágenes organizadas en una carpeta por categoría dentro de `dataset/`. Se utiliza una división de **85% entrenamiento / 15% prueba** por clase.

| Categoría | Imágenes disponibles |
|-----------|----------------------|
| Cartón (cardboard) | 403 |
| Vidrio (glass) | 501 |
| Metal (metal) | 410 |
| Papel (paper) | 594 |
| Plástico (plastic) | 482 |

## Requisitos

- Python 3.8 – 3.12
- Dependencias:

  ```bash
  pip install tensorflow streamlit pillow numpy scikit-learn opencv-python
  ```

## Uso

### Ejecutar la aplicación web

El repositorio ya incluye el modelo entrenado, así que se puede usar directamente:

```bash
streamlit run main.py
```

Se abrirá una página en el navegador. Sube una imagen de un material reciclable y la aplicación mostrará la categoría predicha.

### Reentrenar el modelo (opcional)

Para entrenar la red desde cero con el dataset incluido:

```bash
python IAEntrenamiento.py
```

El script entrena la CNN, guarda el modelo en `model/IAReciclaje.keras` e imprime la matriz de confusión y el reporte de clasificación sobre el conjunto de prueba.

## Estructura del proyecto

```
.
├── IAEntrenamiento.py    # Script de entrenamiento de la CNN
├── main.py               # Aplicación web en Streamlit
├── model/
│   └── IAReciclaje.keras # Modelo entrenado
└── dataset/              # Imágenes de entrenamiento, una carpeta por categoría
    ├── cardboard/
    ├── glass/
    ├── metal/
    ├── paper/
    └── plastic/
```

## Autor

**Luis Fernando Chunwa Chow Cheung**
Estudiante de Ingeniería en Computación — Universidad Rafael Urdaneta
GitHub: [@LuisChow](https://github.com/LuisChow)
