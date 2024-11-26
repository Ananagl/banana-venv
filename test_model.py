import cv2
import numpy as np
from tensorflow.keras.models import load_model
from config import categories  # Asegúrate de tener las categorías definidas en tu archivo de configuración

# Cargar el modelo entrenado
model = load_model("C:\\Users\\Ana\\PycharmProjects\\PythonProject\\.venv\\final_model.keras")

# Función para cargar y preprocesar la imagen de prueba
def preprocess_image(image_path):
    img = cv2.imread(image_path)
    img = cv2.resize(img, (224, 224))  # Redimensionar a 224x224
    img = img / 255.0  # Normalizar la imagen
    img = np.expand_dims(img, axis=0)  # Añadir dimensión de lote
    return img

# Función para hacer la predicción
def predict_image(model, image_path):
    img = preprocess_image(image_path)
    prediction = model.predict(img)
    predicted_class = np.argmax(prediction)  # Obtener la clase con la mayor probabilidad
    predicted_label = list(categories.values())[predicted_class]  # Mapeo de la etiqueta
    return predicted_label

# Ruta de la imagen de prueba
test_image_path = 'C:/Users/Nagle/Desktop/bananamanchas.webp'  # Reemplaza con la ruta de tu imagen

# Realizar la predicción
predicted_label = predict_image(model, test_image_path)

print(f"Predicción para la imagen {test_image_path}: {predicted_label}")
