### Script para ejecutar las funciones de preprocesamiento ###

import pandas as pd
import numpy as np
from PIL import Image
import pydicom
from config import PROCESSED_DIR, RAW_DATA_DIR
from preprocessing import crop_breast, normalize_image, resize_with_padding


#Configuración
CSV_PATH = PROCESSED_DIR / "dataset.csv"
IMAGES_DIR = PROCESSED_DIR / "images"
PREVIEWS_DIR = PROCESSED_DIR / "previews"

#Crear carpetas para guardar las imágenes de train y test procesadas
for split in ["train", "val", "test"]:
    (IMAGES_DIR / split).mkdir(parents=True, exist_ok=True)
    (PREVIEWS_DIR / split).mkdir(parents=True, exist_ok=True)

# Cargar csv
df = pd.read_csv(CSV_PATH)
print("Número total de imágenes:", len(df))
MAX_IMAGES = len(df)

# Añadir al csv la columna processed_image para tener la ruta de la imagen procesada
if "processed_image" not in df.columns:
    df["processed_image"] = ""

# Procesamiento de imágenes
num_images = min(MAX_IMAGES, len(df))
for index in range(num_images):
    # Información de la fila
    row = df.iloc[index]
    split = row["split"]
    print("Índice:", index)
    print("Paciente:", row["patient_id"])
    print("Etiqueta:", row["label"])
    print("Split:", split)

    # Cargar DICOM
    image_path = RAW_DATA_DIR / row["image file path"]
    dicom = pydicom.dcmread(image_path)
    image = dicom.pixel_array

    # Recorte
    cropped, mask = crop_breast(image)

    # Normalización
    normalized = normalize_image(cropped, mask)

    # Resize + padding
    resized = resize_with_padding(normalized, target_size=(512, 512))
    resized = np.clip(resized, 0.0, 1.0)

    # Comprobaciones
    assert resized.shape == (512, 512)
    assert resized.dtype == np.float32
    assert resized.min() >= 0.0
    assert resized.max() <= 1.0

    # Nombre de la imagen procesada (tanto archivo de datos como imagen png)
    image_id = f"{index:06d}"
    npy_path = (IMAGES_DIR / split / f"{image_id}.npy")
    png_path = (PREVIEWS_DIR / split / f"{image_id}.png")

    # Guardar archivos .npy y .png
    np.save(npy_path, resized)
    png_image = (resized * 255).clip(0, 255).astype(np.uint8)
    Image.fromarray(png_image).save(png_path)

    # Añadir ruta en el CSV
    df.loc[index, "processed_image"] = str(npy_path.relative_to(PROCESSED_DIR))

    print("\nGuardado correctamente:")
    print("NPY:", npy_path)
    print("PNG:", png_path)
    print("Shape:", resized.shape)
    print("Dtype:", resized.dtype)
    print("Min:", resized.min())
    print("Max:", resized.max())

df.to_csv(CSV_PATH, index=False)
print("PROCESAMIENTO TERMINADO")