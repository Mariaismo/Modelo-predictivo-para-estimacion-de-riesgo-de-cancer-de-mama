### Preparar el dataset final con ROIs a partir de los CSV, las imágenes DICOM y las ROIs###

from pathlib import Path
import gc
import cv2
import numpy as np
import pandas as pd
import pydicom
from preprocessing import create_breast_mask,get_largest_component,normalize_image, find_binary_roi

RAW_DIR = Path("../data/raw")
PROCESSED_DIR = Path("../data/processed")
DATASET_CSV = PROCESSED_DIR / "dataset.csv"
ORIGINAL_CSVS = [
    Path("../data/metadata/mass_case_description_train_set.csv"),
    Path("../data/metadata/mass_case_description_test_set.csv"),
    Path("../data/metadata/calc_case_description_train_set.csv"),
    Path("../data/metadata/calc_case_description_test_set.csv"),
]
OUTPUT_IMAGES = PROCESSED_DIR / "images_roi_full"
OUTPUT_PREVIEWS = PROCESSED_DIR / "previews_roi_full"
OUTPUT_METADATA = PROCESSED_DIR / "roi_dataset.csv"

ROI_MARGIN = 200
TARGET_SIZE = 512
CHECKPOINT_EVERY = 25

def save_metadata(metadata_by_path):
    """
    Guarda el metadata actual en roi_dataset.csv.
    """

    if not metadata_by_path:
        return

    metadata_df = pd.DataFrame(list(metadata_by_path.values()))

    metadata_df = metadata_df.sort_values(
        by=[
            "split",
            "patient_id",
            "case_folder",
            "roi_index"
        ]
    ).reset_index(drop=True)

    metadata_df.to_csv(OUTPUT_METADATA, index=False)


# Cargar dataset actual (creado con la primera versión de preprocesamiento) y contar cuántas imágenes hay en cada conjunto
dataset = pd.read_csv(DATASET_CSV)
print(dataset["split"].value_counts())

# Cargar dataset originales y unificarlos en un dataframe para tener la info de las ROI
original_dfs = []
for csv_path in ORIGINAL_CSVS:
    df = pd.read_csv(csv_path)
    original_dfs.append(df)
original = pd.concat(original_dfs, ignore_index=True)

# Seleccionar el nombre de cada caso para poder relacionar los datos de dataset.csv con las ROI de los csv originales
original["case_folder"] = (
    original["image file path"]
    .astype(str)
    .str.replace("\\", "/", regex=False)
    .str.split("/")
    .str[0]
)
dataset["case_folder"] = (
    dataset["image file path"]
    .astype(str)
    .str.replace("\\", "/", regex=False)
    .str.split("/")
    .str[0]
)

# Eliminar filas duplicadas para no procesar 2 veces la misma ROI de la misma mamografía
original = original.drop_duplicates(
    subset=[
        "image file path",
        "ROI mask file path"
    ]
).reset_index(drop=True)

print()
print("Anotaciones originales:",len(original)) #Cuántos casos quedan después de elimiar los duplicados

# Comprobar que todas las mamografías que hay en dataset.csv tienen una ROI de los csv originales asociada
dataset_cases = set(dataset["case_folder"])
original_cases = set(original["case_folder"])
missing_cases = dataset_cases - original_cases

print("Casos del dataset sin ROI:", len(missing_cases))

if missing_cases:
    print(list(missing_cases)[:20])
    raise RuntimeError( "Hay mamografías sin correspondencia en los CSV originales.")

# Nuevas carpetas para guardar los resultados
OUTPUT_IMAGES.mkdir(parents=True, exist_ok=True)
OUTPUT_PREVIEWS.mkdir(parents=True, exist_ok=True)

# Tabla auxiliar para encontrar la info de cada caso más rápido utilizando como índice de la tabla el nombre del caso
dataset_by_case = (
    dataset
    .drop_duplicates("case_folder")
    .set_index("case_folder")
)

metadata_by_path = {}

# Comprueba si dataset.csv existe y qué ROIs han sido procesadas
if OUTPUT_METADATA.exists():
    print()
    print("Se ha encontrado un roi_dataset.csv anterior.")
    old_metadata = pd.read_csv(OUTPUT_METADATA)

    for record in old_metadata.to_dict(orient="records"): #cada fila se convierte en un diccionario, metadata_by_path pasa a ser una copia de roi_dataset.csv
        metadata_by_path[record["processed_image"]] = record #se asigna como clave la ruta de la imagen

    print("Registros recuperados:", len(metadata_by_path))

# Juntar las ROIs que pertenecen a un mismo caso
groups = list(original.groupby("case_folder"))
total_cases = len(groups)
processed_cases = 0
failed_cases = 0

print()
print("Casos totales:", total_cases)

# Procesar casos
for case_folder, case_annotations in groups:
    processed_cases += 1

    # Mostrar progreso
    if (processed_cases == 1 or processed_cases % 25 == 0 or processed_cases == total_cases):
        print(f"\nCasos procesados: "f"{processed_cases}/{total_cases}")

    # Comprobar si el caso del dataset está en la tabla auxiliar
    if case_folder not in dataset_by_case.index:
        print("ERROR: caso no encontrado:",case_folder)
        failed_cases += 1
        continue

    # Obtener los datos del caso que se está procesando
    dataset_row = dataset_by_case.loc[case_folder]
    patient_id = dataset_row["patient_id"]
    split = dataset_row["split"]
    mammogram_label = int(dataset_row["label"])

    # Localizar mamografía
    image_case_dir = (RAW_DIR / case_folder)
    image_files = list(image_case_dir.rglob("*full mammogram images*/*.dcm" )) #busca archivos .dcm dentro de full mammogram images para ese caso concreto

    #Asegurar que hay una sola mamografía para el caso
    if len(image_files) != 1:
        print("\nERROR:", case_folder, "tiene", len(image_files), "mamografías.")
        failed_cases += 1
        continue
    
    image_file = image_files[0]

    # Leer mamografía
    try:
        ds = pydicom.dcmread(image_file) #leer archivo DICOM
        image = ds.pixel_array #extraer los pixeles de la mamografía: pasa a ser un array de NumPy
        del ds #liberar memoria
        gc.collect() #liberar memoria

    except Exception as e:
        print("\nERROR leyendo mamografía:", image_file)
        print("Detalle:", e)
        failed_cases += 1
        continue

    # Recorrer las ROIs que hay en la mamografía que se está evaluando
    for roi_index, (_, annotation) in enumerate(case_annotations.iterrows(), start=1): #obtener los datos de cada ROI (guardados en annotation) y asignar un índice consecutivo a cada ROI
        roi_relative_path = str(annotation[ "ROI mask file path"]) #ruta de la ROI

        # Obtener patología y label de la ROI
        pathology = str(annotation["pathology"]).strip().upper()
        roi_label = (1 if pathology == "MALIGNANT" else 0)

        name = (f"{split}_"f"{patient_id}_" f"{case_folder}_" f"roi_{roi_index:02d}") #nombre para la ROI
        npy_path = (OUTPUT_IMAGES /f"{name}.npy") #path para guardar el .npy que entrará a la red
        preview_path = (OUTPUT_PREVIEWS /f"{name}.png") #path para guardar las visualizaciones
        processed_image = str(npy_path.relative_to(PROCESSED_DIR)).replace("\\", "/") #ruta para guardar en dataset.csv

        # Comprobar si ya fue procesada, si existe el archivo .npy y la visualización, es que ya está procesada
        if (npy_path.exists() and preview_path.exists()):
            print("Ya existe, se omite:", npy_path.name)

            # Si el roi_dataset.csv ya contiene el registro, no hay que modificarlo
            if processed_image not in metadata_by_path:
                metadata_by_path[processed_image] = {
                    "patient_id": patient_id,
                    "case_folder": case_folder,
                    "split": split,
                    "roi_index": roi_index,
                    "pathology": pathology,
                    "roi_label": roi_label,
                    "mammogram_label": mammogram_label,
                    "roi_file": roi_relative_path,
                    "original_width": np.nan,
                    "original_height": np.nan,
                    "final_width": np.nan,
                    "final_height": np.nan,
                    "processed_image": processed_image
                }
            continue

        roi_case_folder = Path(roi_relative_path.replace("\\", "/")).parts[0]
        roi_case_dir = (RAW_DIR / roi_case_folder)

        # Encontrar máscara ROI
        roi_file, roi = find_binary_roi(roi_case_dir, image.shape)

        if roi_file is None:
            print("\nERROR: no se encontró ROI:", roi_relative_path)
            failed_cases += 1
            continue

        # Bounding box
        try:
            ys, xs = np.where(roi > 0) #coordenadas de la ROI (en la máscara binaria 0:fondo, >0:roi)
            if len(xs) == 0:
                print("\nERROR: ROI vacía:", roi_file)
                del roi
                gc.collect()
                failed_cases += 1
                continue

            # Esquinas de la ROI
            x_min = xs.min()
            x_max = xs.max()
            y_min = ys.min()
            y_max = ys.max()

            # Dimensiones de la ROI
            original_width = (x_max - x_min + 1)
            original_height = (y_max - y_min + 1)

            # Recorte + margen
            crop_x_min = max(0, x_min - ROI_MARGIN) #límite izq: borde izq-200 o 0 para no salir de la imagen
            crop_x_max = min(image.shape[1], x_max + ROI_MARGIN) #límite dcho: borde dcho+200 o ancho total para no salir de la imagen
            crop_y_min = max(0, y_min - ROI_MARGIN)
            crop_y_max = min(image.shape[0], y_max + ROI_MARGIN)

            cropped = image[crop_y_min:crop_y_max, crop_x_min:crop_x_max] #recortar la mamografía
            cropped_roi = roi[crop_y_min:crop_y_max, crop_x_min:crop_x_max] #recortar la máscara de la lesión

            breast_mask = create_breast_mask(cropped) #máscara de la mama en la mamografía recortada
            breast_mask = get_largest_component(breast_mask) #filtro para seleccionar la zona más grande, que será la mama

            # Resize 
            height, width = cropped.shape
            scale = min( TARGET_SIZE / width, TARGET_SIZE / height)

            new_width = max(1, int(width * scale))
            new_height = max(1, int(height * scale))

            resized = cv2.resize(cropped, (new_width, new_height), interpolation=cv2.INTER_AREA) #redimensionado de la mamografía recortada
            resized_roi = cv2.resize(cropped_roi, (new_width, new_height), interpolation=cv2.INTER_NEAREST) #redimensionado de la máscara de la lesión recortada
            resized_breast_mask = cv2.resize(breast_mask, (new_width, new_height), interpolation=cv2.INTER_NEAREST) #redimensionado de la máscara de la mama recortada

            # Padding
            padded = np.zeros((TARGET_SIZE, TARGET_SIZE), dtype=resized.dtype) #imagen en negro de 512x512 para mamografía recortada
            padded_roi = np.zeros((TARGET_SIZE, TARGET_SIZE), dtype=np.uint8) #imagen en negro de 512x512 para máscara de la lesión recortada
            padded_breast_mask = np.zeros((TARGET_SIZE, TARGET_SIZE), dtype=np.uint8) #imagen en negro de 512x512 para máscara de la mama recortada

            y_offset = ( TARGET_SIZE - new_height) // 2 #cuánto espacio negro dejar en y
            x_offset = ( TARGET_SIZE - new_width) // 2 #cuánto espacio negro dejar en x

            padded[y_offset:y_offset + new_height, x_offset:x_offset + new_width] = resized #colocar mamografía recortada en imagen negra 
            padded_roi[y_offset:y_offset + new_height, x_offset:x_offset + new_width] = resized_roi #colocar máscara de la lesión recortada en imagen negra
            padded_breast_mask[y_offset:y_offset + new_height, x_offset:x_offset + new_width] = resized_breast_mask #colocar máscara de la mama recortada en imagen negra

            # Normalización
            if np.any(padded_breast_mask > 0):
                normalized = normalize_image(padded, padded_breast_mask)
            else:
                raise RuntimeError("No se ha podido crear una máscara de mama para normalizar")

            # Tamaño final de la lesión
            ys_final, xs_final = np.where(padded_roi > 0)

            if len(xs_final) == 0:
                raise RuntimeError("La ROI desapareció después del resize")

            final_width = (xs_final.max()- xs_final.min()+ 1)
            final_height = (ys_final.max()- ys_final.min()+ 1)

            # Guardar NPY para después usarlo en la red
            np.save(npy_path, normalized)

            # Guardar visualización
            preview = cv2.normalize(padded, None,0, 255, cv2.NORM_MINMAX).astype(np.uint8)
            preview = cv2.cvtColor(preview, cv2.COLOR_GRAY2BGR)
            cv2.rectangle(preview, (xs_final.min(), ys_final.min()), (xs_final.max(), ys_final.max()), (255, 0, 0), 2)
            cv2.imwrite(str(preview_path), preview)

            # Guardar metadata
            metadata_by_path[processed_image] = {
                "patient_id": patient_id,
                "case_folder": case_folder,
                "split": split,
                "roi_index": roi_index,
                "pathology": pathology,
                "roi_label": roi_label,
                "mammogram_label": mammogram_label,
                "roi_file": str(roi_file),
                "original_width": original_width,
                "original_height": original_height,
                "final_width": final_width,
                "final_height": final_height,
                "processed_image": processed_image
            }

            print(f"ROI {roi_index}: "f"{original_width}x{original_height} -> "f"{final_width}x{final_height}")

        except Exception as e:
            print("\nERROR procesando ROI:",roi_relative_path)
            print("Detalle:", e)
            failed_cases += 1

        finally:
            # Liberar memoria de esta ROI
            variables_to_delete = [
                "roi",
                "cropped",
                "cropped_roi",
                "breast_mask",
                "resized",
                "resized_roi",
                "resized_breast_mask",
                "padded",
                "padded_roi",
                "padded_breast_mask",
                "normalized",
                "preview",
            ]

            for variable_name in variables_to_delete:
                if variable_name in locals():
                    try:
                        del locals()[variable_name]
                    except Exception:
                        pass
            gc.collect()

    # Liberar mamografía
    del image
    gc.collect()

    # Punto de guardado
    if (processed_cases % CHECKPOINT_EVERY == 0 or processed_cases == total_cases):
        save_metadata(metadata_by_path)
        print("\nCheckpoint guardado")
        print("Registros acumulados:", len(metadata_by_path))

# Guardar resultado final
save_metadata(metadata_by_path)

# Resumen final
print("\n PROCESAMIENTO ROI FINALIZADO")
print("\nCasos procesados:", processed_cases, "/", total_cases)
print("Casos con error:", failed_cases)
print("Crops registrados:", len(metadata_by_path))

if OUTPUT_METADATA.exists():
    final_metadata = pd.read_csv(OUTPUT_METADATA)
    print("\nDistribución por split:")
    print(final_metadata["split"].value_counts())
    print("\nEtiquetas de ROI:")
    print(final_metadata["roi_label"].value_counts().sort_index())
    print("\nMetadata:", OUTPUT_METADATA)