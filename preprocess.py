import json
import cv2
import numpy as np
import os
from config import IMAGE_DIR, ANNOTATIONS_PATH

# Cargar archivo COCO
with open(ANNOTATIONS_PATH, 'r') as f:
    coco_data = json.load(f)

# Mapear categorías
categories = {cat['id']: cat['name'] for cat in coco_data['categories']}
print("Categorías disponibles:", categories)

# Cargar imágenes y etiquetas
def load_data(image_dir, coco_data):
    X = []
    y = []
    for image_info in coco_data['images']:
        img_path = os.path.join(image_dir, image_info['file_name'])

        # Verificar si la imagen existe
        if not os.path.exists(img_path):
            print(f"Advertencia: No se encontró la imagen: {img_path}")
            continue

        # Cargar la imagen
        img = cv2.imread(img_path)
        if img is not None:
            # Procesar las bounding boxes
            for annotation in [a for a in coco_data['annotations'] if a['image_id'] == image_info['id']]:
                x_bbox, y_bbox, w, h = map(int, annotation['bbox'])  # Cambié 'y' por 'y_bbox'
                roi = img[y_bbox:y_bbox+h, x_bbox:x_bbox+w]  # Extraer la región de interés (ROI)
                roi = cv2.resize(roi, (224, 224))  # Redimensionar ROI a 224x224
                X.append(roi)
                y.append(annotation['category_id'])
        else:
            print(f"Advertencia: No se pudo cargar la imagen: {img_path}")

    return np.array(X) / 255.0, np.array(y)

# Cargar datos
X, y = load_data(IMAGE_DIR, coco_data)

# Guardar los datos como archivos numpy
output_dir = r"C:/Users/Nagle/Documents/labeled_banana_images/preprocessed_data/"
os.makedirs(output_dir, exist_ok=True)  # Crear la carpeta si no existe
np.save(os.path.join(output_dir, "X.npy"), X)
np.save(os.path.join(output_dir, "y.npy"), y)

print("Datos cargados: ", X.shape, y.shape)
print(f"Datos guardados en: {output_dir}")
