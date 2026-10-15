### Seleccionar imágenes para demostrar la explicabilidad###

import torch
import numpy as np
import pandas as pd
from pathlib import Path
from model import BreastCancerResNet
from config import PROCESSED_DIR

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = (PROJECT_ROOT / "outputs" / "experiment5_roi" / "best_model_roi.pth")
CSV_PATH = PROCESSED_DIR / "roi_dataset.csv"
OUTPUT_PATH = (PROJECT_ROOT / "outputs" / "experiment5_roi" / "xai_selected_cases.csv")

# Dispositivo
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)

# Modelo
model = BreastCancerResNet().to(device)
state_dict = torch.load(MODEL_PATH, map_location=device)
model.load_state_dict(state_dict)
model.eval()

#Cargar dataset de test
df = pd.read_csv(CSV_PATH)
test_df = df[df["split"] == "test"].reset_index(drop=True)

# Inferencia
results = []
for idx, row in test_df.iterrows():
    image_path = (PROCESSED_DIR / row["processed_image"])
    image = np.load(image_path).astype(np.float32) #cargar imagen en formato NumPy
    image_tensor = (torch.from_numpy(image).unsqueeze(0).unsqueeze(0).to(device))

    with torch.no_grad(): 
        output = model(image_tensor) #predicciones
        probabilities = torch.softmax(output, dim=1) #probabilidad de malignidad

    prob_benign = probabilities[0, 0].item()
    prob_malignant = probabilities[0, 1].item()
    prediction = int(prob_malignant >= 0.5)
    true_label = int(row["roi_label"])

    # Tipo de resultado
    if true_label == 1 and prediction == 1:
        result_type = "TP" #True positive
        confidence = prob_malignant 
    elif true_label == 0 and prediction == 0:
        result_type = "TN" #True negative
        confidence = prob_benign
    elif true_label == 0 and prediction == 1:
        result_type = "FP" #False positive
        confidence = prob_malignant
    else:
        result_type = "FN" #False negative
        confidence = prob_benign

    results.append(
        {
            "test_index": idx,
            "case_folder": row["case_folder"],
            "patient_id": row["patient_id"],
            "processed_image": row["processed_image"],
            "roi_label": true_label,
            "mammogram_label": int(
                row["mammogram_label"]
            ),
            "prob_benign": prob_benign,
            "prob_malignant": prob_malignant,
            "prediction": prediction,
            "result_type": result_type,
            "confidence": confidence
        }
    )

results_df = pd.DataFrame(results)

# Distribución de las predicciones
print("\nDistribución:")
print(results_df["result_type"].value_counts())

# Seleccionar un caso de cada tipo
selected_cases = []

for result_type in ["TP", "TN", "FP", "FN"]:
    candidates = results_df[results_df["result_type"] == result_type].copy()
    if len(candidates) == 0:
        print(f"\nNo hay casos de tipo {result_type}.")
        continue

    # Se elige el caso de mayor confianza
    selected = candidates.sort_values("confidence", ascending=False).iloc[0]
    selected_cases.append(selected)

selected_df = pd.DataFrame(selected_cases).reset_index(drop=True)

#Casos seleccionados
print("Casos seleccionados para XAI")

for _, row in selected_df.iterrows():
    print("Tipo:", row["result_type"])
    print("Test index:", row["test_index"])
    print("Patient ID:", row["patient_id"])
    print("Case folder:", row["case_folder"])
    print("ROI label:", row["roi_label"])
    print("Probabilidad de benigna:", f"{row['prob_benign']:.4f}")
    print("Probabilidad de maligna:", f"{row['prob_malignant']:.4f}")
    print("Predicción:", row["prediction"])
    print("Confianza:", f"{row['confidence']:.4f}")
    print("Imagen:", row["processed_image"])

#Guardar resultados
selected_df.to_csv(OUTPUT_PATH, index=False)
print("\nArchivo guardado en:", OUTPUT_PATH)
