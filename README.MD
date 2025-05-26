# Clasificador de Materiales Reciclables

Este proyecto utiliza una red neuronal convolucional para clasificar imágenes de materiales reciclables en cinco categorías: Cartón, Vidrio, Metal, Papel y Plástico. La interfaz de usuario está construida con Streamlit.

## Requisitos

- Python 3.8 > Python 3.12
- TensorFlow
- Streamlit
- Pillow
- NumPy
- scikit-learn

Puedes instalar las dependencias ejecutando:

```
pip install tensorflow streamlit pillow numpy scikit-learn
```

## Uso

1. **Entrena el modelo (opcional):**  
   Si ya tienes el archivo `model/IAProyecto3.keras`, puedes saltar este paso.  
   Ejecuta el script de entrenamiento para generar el modelo:
   ```
   python IAEntrenamiento.py
   ```

2. **Ejecuta la aplicación Streamlit:**  
   En la terminal, navega a la carpeta del proyecto y ejecuta:
   ```
   streamlit run main.py
   ```

3. **Usa la aplicación:**  
   - Se abrirá una página web en tu navegador.
   - Haz clic en "Sube una imagen" y selecciona una imagen de un material reciclable.
   - El sistema mostrará la imagen y la clase predicha.

## Estructura de carpetas recomendada

```
Proyecto1/
│
├── main.py
├── IAEntrenamiento.py
├── model/
│   └── IAProyecto3.keras
├── dataset/
│   ├── cardboard/
│   ├── glass/
│   ├── metal/
│   ├── paper/
│   └── plastic/
```

## Notas

- Asegúrate de que el archivo del modelo (`IAProyecto3.keras`) esté en la carpeta `model/`.
- Si tienes problemas con las rutas o dependencias, revisa que estés ejecutando los comandos desde la carpeta correcta.