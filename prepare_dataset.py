### Preparar el dataset de mamografías completas a partir de los CSV y las imágenes DICOM ###

import pandas as pd
from config import METADATA_DIR
from config import PROCESSED_DIR
from config import RAW_DATA_DIR
from sklearn.model_selection import train_test_split

def load_csv(filename): #función para leer un csv y mostrar el número de líneas que tiene
    path = METADATA_DIR / filename
    df = pd.read_csv(path)
    print(f"{filename}")
    print(f"Registros: {len(df)}")
    print()
    return df

def aggregate_pathology(values): #función para designar la patología en caso de haber varias
    values = set(values)
    if "MALIGNANT" in values: #tiene prioridad MALIGNANT
        return "MALIGNANT"
    if "BENIGN" in values: #depués BENING
        return "BENIGN"
    if "BENIGN_WITHOUT_CALLBACK" in values: #por último BENING WITHOUT CALLBACK
        return "BENIGN_WITHOUT_CALLBACK"
    return "UNKNOWN"

def aggregate_lesion_type(values): #función para definir el tipo de lesión
    values = sorted(set(values))
    return "|".join(values) #si está duplicado elimina uno, si están los dos tipos escribe mass|calc


def aggregate_original_split(values):#función para decidir si es train o test
    values = sorted(set(values))
    return "|".join(values) #mismo funcionamiento que aggregate_lession_type

mass_train = load_csv("mass_case_description_train_set.csv") # Masas del conjunto de entrenamiento
mass_test = load_csv("mass_case_description_test_set.csv") # Masas del conjunto de prueba
calc_train = load_csv("calc_case_description_train_set.csv") # Calcificaciones del conjunto de entrenamiento
calc_test = load_csv("calc_case_description_test_set.csv") # Calcificaciones del conjunto de prueba

#añadir a los csv columna split para saber si la imagen originalmente pertenece a train o test
mass_train["split"] = "train"
mass_test["split"] = "test"
calc_train["split"] = "train"
calc_test["split"] = "test"

# añadir a los csv columna lession_type para saber si la imagen es de masa o calcificación
mass_train["lesion_type"] = "mass"
mass_test["lesion_type"] = "mass"
calc_train["lesion_type"] = "calc"
calc_test["lesion_type"] = "calc"

# unir los df de los 4 csv en un solo df
dataset = pd.concat(
    [
        mass_train,
        mass_test,
        calc_train,
        calc_test
    ],
    ignore_index=True #crea un índice nuevo para no tener los índices de las tablas separadas
)

#nombre de la carpeta del caso
dataset["case_folder"] = (
    dataset["image file path"]
    .astype(str) #convierte el contenido en texto
    .str.replace("\\", "/", regex=False) #convierte las barras \\ de las rutas en /
    .str.split("/") #separa la ruta por las barras
    .str[0] #se queda con el primer elemento de la ruta que es el nombre de la carpeta que hay en raw
)

#buscar ruta de la mamografía DICOM
def find_dicom_path(case_folder):
    case_dir = RAW_DATA_DIR / case_folder
    if not case_dir.exists():
        print(f"ERROR: No existe la carpeta: {case_folder}")
        return None
    #buscar únicamente carpetas de mamografía
    mammogram_folders = [
        folder
        for folder in case_dir.rglob("*full mammogram images*") #dentro de la ruta de cada caso busca una carpeta que contenga "full mammogram images"
        if folder.is_dir()
    ]

    #recorre las carpetas con mamografías y busca los archivos con extensión .dcm
    mammogram_files = []
    for folder in mammogram_folders: 
        mammogram_files.extend(folder.glob("*.dcm"))

    #si no se encuentra ninguna mamografía
    if len(mammogram_files) == 0:
        print(
            f"ERROR: No se encontró mamografía completa para "
            f"{case_folder}"
        )
        return None

    #si hay más de una mamografía para el mismo caso
    if len(mammogram_files) > 1:
        print(
            f"ERROR: {case_folder} contiene "
            f"{len(mammogram_files)} mamografías completas."
        )
        return None

    return mammogram_files[0].relative_to(RAW_DATA_DIR).as_posix() #devuelve la ruta relativa de la mamografía desde la carptea raw con /

print("Buscando imágenes DICOM...")

case_folders = dataset["case_folder"].unique()

dicom_map = {}

#recorre los casos y va generanco un diccionario en el que asigna la ruta de la mamografía a cada caso
for i, case_folder in enumerate(case_folders, start=1):
    dicom_map[case_folder] = find_dicom_path(case_folder)
    if i % 100 == 0: #solamente para mostrar el proceso en la terminal
        print(f"Procesadas {i}/{len(case_folders)} carpetas")

dataset["image file path"] = dataset["case_folder"].map(dicom_map) #se sustituye la ruta que había en el csv por la ruta real

#comprobar que todas las imágenes existen, comprueba que ninguna es None
missing = dataset["image file path"].isna().sum()

print("COMPROBACIÓN DE IMÁGENES")
print(f"Total de anotaciones: {len(dataset)}")
print(f"Imágenes no encontradas: {missing}")
print()

#avisa en caso de que alguna imagen no exista
if missing > 0:
    print("Casos sin imagen:")
    print(
        dataset.loc[
            dataset["image file path"].isna(),
            ["case_folder", "patient_id"]
        ].drop_duplicates()
    )
    raise RuntimeError(
        "Hay imágenes DICOM que no se han podido localizar."
    )

print("Columnas disponibles antes de agrupar:")
print(dataset.columns.tolist())
dataset_final = (
    dataset
    .groupby("image file path", as_index=False) #se agrupan las mamografías por rutas, puede haber distintos casos apuntando a la misma mamografía
    .agg( #para esas agrupaciones hay que seleccionar los valores de las columnas
        patient_id=("patient_id", "first"), #en el caso del paciente nos quedamos con el primero porque es el mismo
        pathology=("pathology", aggregate_pathology), #para seleccionar la patología se usa la función aggrgate_pathology
        lesion_type=("lesion_type", aggregate_lesion_type), #para seleccionar el tipo de lesión se usa la función aggregate_lesion_type
        original_split=("split", aggregate_original_split), #para seleccionar si es de train o test se usa la función aggregate_original_split
        case_folder=("case_folder", "first")
    )
)

#crear etiqueta binaria
dataset_final["label"] = (
    dataset_final["pathology"]
    .map(
        {
            "MALIGNANT": 1,
            "BENIGN": 0,
            "BENIGN_WITHOUT_CALLBACK": 0
        }
    )
)

#comprobar que no haya etiquetas desconocidas
unknown_labels = dataset_final["label"].isna().sum()
if unknown_labels > 0:
    print("ERROR: Hay etiquetas que no se han podido convertir.")
    print(
        dataset_final[
            dataset_final["label"].isna()
        ]["pathology"].value_counts()
    )
    raise RuntimeError(
        "Existen valores de pathology no contemplados."
    )

#crear split a nivel de paciente
print()
print("Creando train/test a nivel de paciente...")

# Obtener una etiqueta por paciente: si un paciente tiene alguna mamografía maligna, el paciente se considera maligno para estratificar.
patient_labels = (
    dataset_final
    .groupby("patient_id")["label"]
    .max()
    .reset_index()
)

#Dividir en conjuntos train y test
train_patients, test_patients = train_test_split(
    patient_labels["patient_id"],
    test_size=0.20,
    random_state=42,
    stratify=patient_labels["label"]
)

train_patients = set(train_patients)
test_patients = set(test_patients)

dataset_final["split"] = dataset_final["patient_id"].apply(
    lambda patient:
    "train" if patient in train_patients else "test"
)

#comprobar que no hay pacientes en ambos conjuntos
train_patient_set = set(
    dataset_final.loc[
        dataset_final["split"] == "train",
        "patient_id"
    ]
)

test_patient_set = set(
    dataset_final.loc[
        dataset_final["split"] == "test",
        "patient_id"
    ]
)

overlap = train_patient_set.intersection(test_patient_set)

if len(overlap) > 0:
    print("ERROR: Hay pacientes en train y test:")
    print(overlap)
    raise RuntimeError("Se ha detectado data leakage entre train y test.")

#eliminar columnas auxiliares
dataset_final = dataset_final[
    [
        "patient_id",
        "image file path",
        "pathology",
        "label",
        "lesion_type",
        "original_split",
        "split"
    ]
]

#ordenar el dataset
dataset_final = dataset_final.sort_values(
    by=["split", "patient_id", "image file path"]
).reset_index(drop=True)

# mostrar estadísticas finales
print()
print("DATASET FINAL")
print(f"Anotaciones originales: {len(dataset)}")
print(f"Mamografías únicas: {len(dataset_final)}")
print(f"Pacientes únicos: {dataset_final['patient_id'].nunique()}")
print()
print("Distribución de etiquetas:")
print(dataset_final["label"].value_counts().sort_index())
print()
print("Distribución de pathology:")
print(dataset_final["pathology"].value_counts())
print()
print("Distribución train/test:")
print(dataset_final["split"].value_counts())
print()
print("Pacientes train:", len(train_patient_set))
print("Pacientes test:", len(test_patient_set))
print()
print("Comprobación de pacientes compartidos:")
print("Pacientes en ambos conjuntos:", len(overlap))


#guardar dataset final
PROCESSED_DIR.mkdir(exist_ok=True)
output_path = PROCESSED_DIR / "dataset.csv"
dataset_final.to_csv(output_path, index=False)
print()
print("DATASET GENERADO CORRECTAMENTE")
print(f"Ruta: {output_path}")
print(f"Muestras finales: {len(dataset_final)}")