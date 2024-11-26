import json
import os
import cv2
import matplotlib.pyplot as plt
from config import IMAGE_DIR, ANNOTATIONS_PATH

# Cargar archivo COCO
with open(ANNOTATIONS_PATH, 'r') as f:
    coco_data = json.load(f)

# Mapear categorías
categories = {cat['id']: cat['name'] for cat in coco_data['categories']}
print("Categorías disponibles:", categories)

# Buscar y mostrar una imagen de cada categoría
for category_id, category_name in categories.items():
    if category_name != 'Banana':  # Evitar la categoría 'Banana', ya que no tiene información de grado
        # Buscar la primera imagen que tenga esta categoría
        for annotation in coco_data['annotations']:
            if annotation['category_id'] == category_id:
                image_info = next(img for img in coco_data['images'] if img['id'] == annotation['image_id'])
                img_path = os.path.join(IMAGE_DIR, image_info['file_name'])
                img = cv2.imread(img_path)

                if img is not None:
                    # Mostrar la imagen con el grado correspondiente
                    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
                    plt.title(f"Grado: {category_name}")
                    plt.axis('off')  # Quitar los ejes
                    plt.show()
                    break