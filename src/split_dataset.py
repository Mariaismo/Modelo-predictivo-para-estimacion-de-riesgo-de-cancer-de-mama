### Dividir el dataset de train en conjuntos de train y validation"

import pandas as pd
from sklearn.model_selection import train_test_split
from config import PROCESSED_DIR

# Cargar dataset
df = pd.read_csv(PROCESSED_DIR / "dataset.csv")

# Seleccionar únicamente con los pacientes de train
train_df = df[df["split"] == "train"].copy()

# Obtener pacientes únicos y su etiqueta
patients = (train_df.groupby("patient_id")["label"].max().reset_index())

# Separar pacientes en train y validation
train_patients, val_patients = train_test_split(
    patients,
    test_size=0.20,
    random_state=42,
    stratify=patients["label"]
)

# IDs de pacientes de cada grupo
train_patient_ids = set(train_patients["patient_id"])
val_patient_ids = set(val_patients["patient_id"])

# Actualizar el split
df.loc[
    (df["split"] == "train") &
    (df["patient_id"].isin(val_patient_ids)),
    "split"
] = "val"

# Guardar
df.to_csv(PROCESSED_DIR / "dataset.csv", index=False)

print("Pacientes:")
print(df.groupby("split")["patient_id"].nunique())

print("\nMamografías:")
print(df["split"].value_counts())