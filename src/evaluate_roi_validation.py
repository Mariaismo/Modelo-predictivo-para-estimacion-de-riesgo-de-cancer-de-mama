### Evalúa el conjunto de validación para las ROI###

import torch
import pandas as pd
from sklearn.metrics import confusion_matrix
from model import BreastCancerResNet
from dataloader_roi import create_roi_dataloaders
from evaluate_roi_test import calculate_metrics

MODEL_PATH = "../outputs/experiment5_roi/best_model_roi.pth"

if __name__ == "__main__":
    # Dispositivo
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Dispositivo:", device)

    # Cargar dataset de validación
    _, val_loader, _ = create_roi_dataloaders()
    print("ROI de validation:", len(val_loader.dataset))

    # Cargar modelo seleccionado
    model = BreastCancerResNet()
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model = model.to(device)
    model.eval()

    # Predicciones de todas las ROI
    records = []
    with torch.no_grad():
        for images, roi_labels, case_folders, mammogram_labels in val_loader:
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

    # DataFrame de ROI
    roi_predictions = pd.DataFrame(records)

    # Agrupar por mamografía: la probabilidad de malignidad de lamamografía es la máxima probabilidadde sus ROI
    mammogram_predictions = (
        roi_predictions.groupby("case_folder", as_index=False)
        .agg(
            mammogram_label=("mammogram_label", "first"),
            mammogram_probability=("roi_probability", "max"),
            num_roi=("roi_probability", "count")
        )
    )

    # Predicción con threshold 0.5
    mammogram_predictions["prediction"] = (mammogram_predictions["mammogram_probability"] >= 0.5).astype(int)

    # Arrays    
    y_true = mammogram_predictions["mammogram_label"].values
    y_pred = mammogram_predictions["prediction"].values
    y_prob = mammogram_predictions["mammogram_probability"].values


    # Métricas
    metrics = calculate_metrics(y_true, y_pred, y_prob)

    # Mostrar resultados
    print("\nResultados de la evaluación sobre validación")
    print("\nMamografías evaluadas:", len(mammogram_predictions))
    print("ROI evaluadas:", len(roi_predictions))
    print("\nAccuracy:", f"{metrics['accuracy']:.4f}")
    print("Precision:", f"{metrics['precision']:.4f}")
    print("Sensitivity:", f"{metrics['sensitivity']:.4f}")
    print("Specificity:", f"{metrics['specificity']:.4f}")
    print("F1:", f"{metrics['f1']:.4f}")
    print("ROC-AUC:", f"{metrics['auc:.4f']}")
    print("\nConfusion matrix:",confusion_matrix(y_true, y_pred))

    # Guardar predicciones
    output_path = ("../outputs/experiment5_roi/validation_mammogram_predictions.csv")
    mammogram_predictions.to_csv(output_path, index=False)
    print("\nPredicciones guardadas en:", output_path)
