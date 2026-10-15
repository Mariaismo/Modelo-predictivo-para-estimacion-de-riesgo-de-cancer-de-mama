### Prueba exploratoria para comprobar cuánto se reduce una lesión al aplicar el preprocesamiento de la mamografía completa ###

import numpy as np
import pandas as pd
import pydicom
from pathlib import Path
from config import IMAGE_SIZE
from preprocessing import crop_breast, resize_with_padding, create_breast_mask, get_largest_component, get_bounding_box

# Cargar el primer caso del csv
csv_path = Path("../data/metadata/calc_case_description_test_set.csv")
df = pd.read_csv(csv_path)
row = df.iloc[0]
print("Paciente:", row["patient_id"])
print("Lesión:", row["abnormality type"])
print("Pathology:", row["pathology"])

# Localizar los archivos DICOM de la mamografía y la ROI
raw_dir = Path("../data/raw")
image_case_dir = raw_dir / Path(row["image file path"]).parts[0]
roi_case_dir = raw_dir / Path(row["ROI mask file path"]).parts[0]
image_file = list(image_case_dir.rglob("*.dcm"))[0]
roi_files = list(roi_case_dir.rglob("*.dcm"))

# Busca la máscara binaria 
roi_file = None
for f in roi_files: #recorre todas las ROI encontradas
    test = pydicom.dcmread(f).pixel_array #extrae matriz de píxeles de la ROI analizada
    if test.dtype == np.uint8 and set(np.unique(test)).issubset({0, 255}): #comprueba que es una máscra binaria con valores 0 y 255
        roi_file = f #si cumple las condiciones, la guarda
        break
if roi_file is None: #si ningún archivo cumple las condiciones
    raise RuntimeError("No se ha encontrado la máscara ROI binaria.")

print("\nMamografía:")
print(image_file)
print("\nMáscara ROI:")
print(roi_file)

# Leer mamografía y ROI en forma de matriz NumPy y transforma la máscara a binaria
image = pydicom.dcmread(image_file).pixel_array #lee los pixeles de la mamografía
roi_mask = pydicom.dcmread(roi_file).pixel_array #lee los pixeles de la ROI
roi_mask = (roi_mask > 0).astype(np.uint8) * 255 #coge los pixeles>0, devuelve True (False para el resto), lo convierte a 1 y multiplica por 255
print("\nImagen original:", image.shape)
print("Máscara ROI:", roi_mask.shape)

# Analiza el tamaño de la lesión original
ys, xs = np.where(roi_mask > 0)
x_min = xs.min()
x_max = xs.max()
y_min = ys.min()
y_max = ys.max()
original_width = x_max - x_min + 1
original_height = y_max - y_min + 1
print("\nLESIÓN ORIGINAL")
print("Width:", original_width)
print("Height:", original_height)
print("Área:", np.sum(roi_mask > 0))

# Recorte de la mamografía con crop_breast()
cropped_image, _ = crop_breast(image)
print("\nImagen después de crop_breast():")
print(cropped_image.shape)

# Recorte de 100px por arriba y por abajo en la mamografía y en la ROI para eliminar bordes de la imagen
top_crop = 100
bottom_crop = 100
image_preprocessed = image[top_crop:-bottom_crop, :]
roi_preprocessed = roi_mask[top_crop:-bottom_crop, :]

# Máscara utilizada por nuestro preprocessing
breast_mask = create_breast_mask(image_preprocessed)
largest_mask = get_largest_component(breast_mask)
x, y, w, h = get_bounding_box(largest_mask)

# Tras calcular el bounding box de la mama, deja un margen de 300px
margin_x = 300
margin_y = 300
# Con min y max se evita que ese margen se salga de la imagen
x_min_crop = max(0, x - margin_x)
x_max_crop = min(image_preprocessed.shape[1], x + w + margin_x)
y_min_crop = max(0, y - margin_y)
y_max_crop = min(image_preprocessed.shape[0], y + h + margin_y)

#Se recorta la ROI con ese margen
roi_cropped = roi_preprocessed[y_min_crop:y_max_crop, x_min_crop:x_max_crop].copy()

# En crop_breast() se quita el borde para eliminar el borde blanco, se aplica lo mismo a la ROI
border_remove = 10
if x < image_preprocessed.shape[1] / 2: #si el bounding box empieza antes del centro de la imagen
    roi_cropped = roi_cropped[:, border_remove:] #elimina los 10px del lateral izdo
else:
    roi_cropped = roi_cropped[:, :-border_remove] #elimina los 10px del lateral dcho

print("\nROI después del crop:")
print(roi_cropped.shape)

# Resize de la ROI
roi_512 = resize_with_padding(roi_cropped, target_size=IMAGE_SIZE)

# Tamaño final de la lesión
ys, xs = np.where(roi_512 > 0)
if len(xs) == 0:
    print("\nERROR: la ROI ha desaparecido.")

else:
    final_x_min = xs.min()
    final_x_max = xs.max()
    final_y_min = ys.min()
    final_y_max = ys.max()
    final_width = final_x_max - final_x_min + 1
    final_height = final_y_max - final_y_min + 1
    final_area = np.sum(roi_512 > 0)

    #Resultados
    print("\nRESULTADO FINAL")
    print("\nImagen original:")
    print(image.shape)

    print("\nLesión original:")
    print(f"{original_width} × {original_height} px")

    print("\nImagen después del crop:")
    print(cropped_image.shape)

    print("\nImagen que recibe ResNet:")
    print("512 × 512")

    print("\nLesión en la imagen final:")
    print(f"{final_width} × {final_height} px")

    print("\nÁrea de la lesión en 512×512:")
    print(final_area, "píxeles")

    print("\nPorcentaje del área de la imagen:")
    print(f"{100 * final_area / (512 * 512):.2f}%")