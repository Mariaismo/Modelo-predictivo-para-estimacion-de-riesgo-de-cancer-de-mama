### Evalúa el conjunto de test para las ROI###

import torch
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
from model import BreastCancerResNet
from dataloader_roi import create_roi_dataloaders

MODEL_PATH = ("../outputs/experiment5_roi/best_model_roi.pth")
OUTPUT_PATH = ("../outputs/experiment5_roi/test_mammogram_predictions.csv")

# Métricas
def calculate_metrics(y_true, y_pred, y_prob):
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    specificity = (tn / (tn + fp) if (tn + fp) > 0 else 0.0)

    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "sensitivity": recall_score(y_true, y_pred, zero_division=0),
        "specificity": specificity,
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "roc_auc": roc_auc_score( y_true, y_prob),
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "tp": tp,
    }

if __name__ == "__main__":
    # Dispositivo
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Dispositivo:", device)

    # Cargar dataset de test
    _, _, test_loader = (create_roi_dataloaders())
    print("ROI de test:", len(test_loader.dataset))

    # Cargar modelo seleccionado
    model = BreastCancerResNet()
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model = model.to(device)
    model.eval()

    # Predicciones de todas las ROI
    records = []
    with torch.no_grad():
        for images, roi_labels, case_folders, mammogram_labels in test_loader:
            images = images.to(device)
            outputs = model(images)
            probabilities = torch.softmax(outputs, dim=1) #devuelve la probabilidad de pertenencia a cada clase
            malignant_probabilities = (probabilities[:, 1].cpu().numpy()) #probabilidad de que sea maligno
            mammogram_labels = (mammogram_labels.cpu().numpy())

            for i, case_folder in enumerate(case_folders):
                records.append({
                    "case_folder": case_folder,
                    "mammogram_label": int(mammogram_labels[i]),
                    "roi_probability": float(malignant_probabilities[i])
                })

    # DataFrame ROI
    roi_predictions = pd.DataFrame(records)
    print("\nROI evaluadas:", len(roi_predictions))

    # Agrupar por mamografía: la probabilidad de malignidad de lamamografía es la máxima probabilidadde sus ROI
    mammogram_predictions = (
        roi_predictions
        .groupby("case_folder", as_index=False)
        .agg(
            mammogram_label=("mammogram_label", "first"), #se queda con el nombre del caso de la primer ROI (todas las ROI tienen el mismo nombre)
            mammogram_probability=("roi_probability", "max"),  #se queda con la probabilidad de malignidad más alta
            num_roi=("roi_probability", "count") #número de ROIs de la mamografía analizada
        )
    )

    # Predicción usando threshold 0.5
    mammogram_predictions["prediction"] = (mammogram_predictions["mammogram_probability"] >= 0.5).astype(int)

    # Arrays
    y_true = mammogram_predictions["mammogram_label"].to_numpy()
    y_pred = mammogram_predictions["prediction"].to_numpy()
    y_prob = mammogram_predictions[ "mammogram_probability"].to_numpy()

    # Métricas
    metrics = calculate_metrics(y_true, y_pred, y_prob)

    # Imprimir resultados
    print("\nRESULTADO FINAL DE TEST")
    print("\nMamografías evaluadas:", len(mammogram_predictions))
    print("ROI evaluadas:", len(roi_predictions))
    print("\nAccuracy:", f"{metrics['accuracy']:.4f}")
    print("Precision:", f"{metrics['precision']:.4f}")
    print("Sensitivity:", f"{metrics['sensitivity']:.4f}")
    print("Specificity:", f"{metrics['specificity']:.4f}")
    print("F1:", f"{metrics['f1']:.4f}")
    print("ROC-AUC:", f"{metrics['roc_auc']:.4f}")
    print("\nConfusion matrix:",confusion_matrix(y_true, y_pred))

    # Guardar predicciones
    mammogram_predictions.to_csv(OUTPUT_PATH, index=False)
    print("\nPredicciones guardadas en:")
    print(OUTPUT_PATH)
