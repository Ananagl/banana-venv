import json
import os
import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization, GlobalAveragePooling2D
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split  # Para dividir los datos en entrenamiento y validación
from config import IMAGE_DIR, ANNOTATIONS_PATH

# Cargar el modelo guardado previamente (si existe, si no se entrena desde cero)
model_path = "C:\\Users\\Ana\\PycharmProjects\\PythonProject\\.venv\\final_model.keras"
if os.path.exists(model_path):
    model = load_model(model_path)
    print("Modelo cargado desde", model_path)
else:
    # Si el modelo no existe, crear uno nuevo
    print("No se encontró un modelo guardado. Se creará un modelo nuevo.")
    model = None

# Cargar archivo COCO
with open(ANNOTATIONS_PATH, 'r') as f:
    coco_data = json.load(f)

# Mapear categorías
categories = {cat['id']: cat['name'] for cat in coco_data['categories']}
print("Categorías disponibles:", categories)

# Cargar datos preprocesados
X = np.load('C:/Users/Nagle/Documents/labeled_banana_images/preprocessed_data/X.npy')
y = np.load('C:/Users/Nagle/Documents/labeled_banana_images/preprocessed_data/y.npy')

# Asegúrate de que las etiquetas sean enteros
y = y.astype(int)

# Dividir los datos en entrenamiento y validación (80% entrenamiento, 20% validación)
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Si el modelo no estaba previamente cargado, crear uno nuevo
if model is None:
    def create_model(input_shape):
        model = Sequential([
            Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
            BatchNormalization(),
            MaxPooling2D(pool_size=(2, 2)),

            Conv2D(64, (3, 3), activation='relu'),
            BatchNormalization(),
            MaxPooling2D(pool_size=(2, 2)),

            Conv2D(128, (3, 3), activation='relu'),
            BatchNormalization(),
            MaxPooling2D(pool_size=(2, 2)),

            Flatten(),
            Dense(128, activation='relu'),
            Dropout(0.5),
            Dense(len(categories), activation='softmax')  # Número de clases
        ])
        model.compile(optimizer=Adam(), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        return model

    # Crear el modelo
    model = create_model(X.shape[1:])

# Definir directorio para guardar el modelo
checkpoint_dir = 'C:/Users/Nagle/Documents/BananaRipenessProject/.venv/checkpoints/'
os.makedirs(checkpoint_dir, exist_ok=True)  # Crear la carpeta de checkpoints si no existe

# Crear el callback para guardar el modelo cuando se mejora la validación
checkpoint_callback = ModelCheckpoint(
    filepath=os.path.join(checkpoint_dir, 'best_model.keras'),  # Guarda el mejor modelo con extensión .keras
    monitor='val_loss',  # Se monitorea la pérdida de validación
    save_best_only=True,  # Solo guarda el mejor modelo
    save_weights_only=False,  # Guarda todo el modelo
    verbose=1  # Para ver los mensajes en la consola
)

# Agregar el callback EarlyStopping para detener el entrenamiento si no mejora
early_stopping = EarlyStopping(
    monitor='val_loss',
    patience=5,  # Detener después de 5 épocas sin mejora
    restore_best_weights=True,  # Restaurar los mejores pesos
    verbose=1
)

# Usar ImageDataGenerator para realizar Data Augmentation
datagen = ImageDataGenerator(
    rotation_range=30,  # Rotar imágenes aleatoriamente
    width_shift_range=0.2,  # Desplazar horizontalmente
    height_shift_range=0.2,  # Desplazar verticalmente
    shear_range=0.2,  # Cortar aleatoriamente las imágenes
    zoom_range=0.2,  # Aplicar zoom aleatorio
    horizontal_flip=True,  # Voltear horizontalmente
    fill_mode='nearest'  # Rellenar los píxeles vacíos después de las transformaciones
)

# Ajustar el generador de datos al conjunto de entrenamiento
datagen.fit(X_train)

# Reentrenar el modelo (o continuar entrenamiento si ya fue cargado)
history = model.fit(
    datagen.flow(X_train, y_train, batch_size=32),  # Usamos el generador de datos con augmentation
    epochs=60,  # Incrementar el número de épocas
    validation_data=(X_val, y_val),  # Usamos el conjunto de validación
    callbacks=[checkpoint_callback, early_stopping],  # Usar los callbacks
    verbose=1  # Muestra detalles del entrenamiento
)

# Guardar el modelo final después de todas las épocas
final_model_path = 'C:/Users/Nagle/Documents/BananaRipenessProject/.venv/final_model.keras'
model.save(final_model_path)  # Guardar el modelo en formato .keras

print("Entrenamiento completado y modelo guardado.")
