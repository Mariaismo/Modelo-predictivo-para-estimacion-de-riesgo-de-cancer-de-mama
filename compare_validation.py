### Compara los cinco experimentos utilizando el conjunto de validación ###

from pathlib import Path
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
from model import BreastCancerResNet
from dataloader import create_dataloaders

BASE_DIR = Path("../ResultadosTrainings")
EXPERIMENT5_DIR = Path("../outputs/experiment5_roi")
MODELS = {
    "Experimento 1 - Baseline":BASE_DIR / "Train1" / "best_model.pth",
    "Experimento 2 - Augmentation":BASE_DIR / "Train2" / "best_model.pth",
    "Experimento 3 - Augmentation + Weight Decay":BASE_DIR / "Train3" / "best_model.pth",
    "Experimento 4 - Augmentation + WD + LR":BASE_DIR / "Train4" / "best_model.pth",
}
ROI_VALIDATION_FILE = (EXPERIMENT5_DIR /"validation_mammogram_predictions.csv")

# Cálculo de métricas
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
        "roc_auc": roc_auc_score(y_true, y_prob),
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "tp": tp,
    }

# Evaluación para experimentos 1-4 (mamografía completa)
def evaluate_standard_model(model_path, val_loader, device):
    model = BreastCancerResNet()
    state_dict = torch.load(model_path, map_location=device) #cargar pesos del modelo
    model.load_state_dict(state_dict)
    model = model.to(device)
    model.eval() #activar modo evaluación

    all_labels = []
    all_probabilities = []

    with torch.no_grad():
        for images, labels in val_loader: #recorre el conjunto de validación
            images = images.to(device)
            outputs = model(images)
            probabilities = torch.softmax(outputs, dim=1) #probabilidades de maligno y benigno para cada caso de validación 
            malignant_probabilities = (probabilities[:, 1].cpu().numpy()) #probabilidad de malignidad
            all_probabilities.extend(malignant_probabilities) 
            all_labels.extend(labels.numpy())

    y_true = np.array(all_labels) #todas etiquetas reales
    y_prob = np.array(all_probabilities) #todas las probabilidades de malignidad
    y_pred = (y_prob >= 0.5).astype(int) #si la probabilidad es >0.5, es maligno
    metrics = calculate_metrics(y_true, y_pred, y_prob)
    return metrics

if __name__ == "__main__":

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Dispositivo:", device)

    # Cargar el conjunto de validación solamente
    _, val_loader, _ = create_dataloaders()
    print("Mamografías de validation:", len(val_loader.dataset))
    results = []

    # Experimentos 1-4
    for experiment_name, model_path in MODELS.items():
        print("Evaluando:", experiment_name)
        print("Modelo:", model_path)
        metrics = evaluate_standard_model(model_path, val_loader, device)
        metrics["experiment"] = (experiment_name)
        results.append(metrics)

        print(f"Accuracy: {metrics['accuracy']:.4f}")
        print(f"Precision: {metrics['precision']:.4f}")
        print(f"Sensitivity: {metrics['sensitivity']:.4f}")
        print(f"Specificity: {metrics['specificity']:.4f}")
        print(f"F1: {metrics['f1']:.4f}")
        print(f"ROC-AUC: {metrics['roc_auc']:.4f}")

    # Experimento 5 (ROI)
    print("\nCargando resultados del Experimento 5")
    roi_predictions = pd.read_csv(ROI_VALIDATION_FILE)
    y_true = roi_predictions["mammogram_label"].to_numpy()
    y_prob = roi_predictions["mammogram_probability"].to_numpy()
    y_pred = (y_prob >= 0.5).astype(int)
    roi_metrics = calculate_metrics(y_true, y_pred, y_prob)
    roi_metrics["experiment"] = ("Experimento 5 - ROI")
    results.append(roi_metrics)

    print(f"Accuracy: {roi_metrics['accuracy']:.4f}")
    print(f"Precision: {roi_metrics['precision']:.4f}")
    print(f"Sensitivity: {roi_metrics['sensitivity']:.4f}")
    print(f"Specificity: {roi_metrics['specificity']:.4f}")
    print(f"F1: {roi_metrics['f1']:.4f}")
    print(f"ROC-AUC: {roi_metrics['roc_auc']:.4f}")

    # Guardar resultados
    results_df = pd.DataFrame(results)
    results_df = results_df[["experiment", "accuracy", "precision", "sensitivity", "specificity", "f1", "roc_auc", "tn", "fp", "fn", "tp"]]
    output_path = (EXPERIMENT5_DIR / "validation_experiments_comparison.csv")
    results_df.to_csv(output_path, index=False)
    print("\nResultados guardados en:")
    print(output_path)

