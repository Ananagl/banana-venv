-------------------------------------------------------------------------------------------------------------------
Dataset usado: https://www.kaggle.com/datasets/lucianon/banana-ripeness-dataset
Crear un proyecto Python de entorno virtual, dentro de la carpeta .venv crear una carpeta llamada Backend y Dataset
-------------------------------------------------------------------------------------------------------------------
En Backend va el archivo app.py (corre el entorno virtual en http://127.0.0.1:5000)
En Dataset va el archivo annotations.coco.json (anotaciones de cada imagen del dataset)
En Scripts van los archivos preprocess.py (preprocesa las imagenes para entrenar el modelo),
train_model.py (entrena el modelo)
config.py (asigna las rutas)
test_model.py (testea el modelo con una imagen seleccionada)
degrees_test.py (muestra la primera imagen de cada grado del dataset)
-------------------------------------------------------------------------------------------------------------------
*Hay que cambiar las rutas de los archivos para que coincidan*
