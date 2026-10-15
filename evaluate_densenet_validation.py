### Evaluación del conjunto de validación para datos de entrada de ROIs con modelo DenseNet-121###

import torch
import pandas as pd
from sklearn.metrics import confusion_matrix
from config import PROCESSED_DIR
from dataloader_roi import create_roi_dataloaders
from model_densenet import BreastCancerDenseNet
from evaluate_roi_test import calculate_metrics

OUTPUT_PATH = ("../outputs/experiment5_roi/validation_mammogram_predictions.csv")

if __name__ == "__main__":
    #Dispositivo
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print("Device:", device)
    
        #Datos para evaluación
        _, val_loader, _ = create_roi_dataloaders()
    
        # Modelo
        model = BreastCancerDenseNet().to(device)
        model_path = (PROCESSED_DIR.parent.parent / "outputs" / "experiment6_densenet" / "best_model_densenet.pth")
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.eval()
    
        # Inferencia
        results = []
    
        with torch.no_grad():
            for images, roi_labels, case_folders, mammogram_labels in val_loader:
                images = images.to(device)
                outputs = model(images)
                probabilities = torch.softmax(outputs, dim=1)[:, 1] #devuelve la probabilidad de pertenencia a cada clase
                probabilities = probabilities.cpu().numpy() #diagnóstico del modelo
                roi_labels = roi_labels.numpy()
                mammogram_labels = mammogram_labels.numpy()
    
                for i in range(len(case_folders)): #recorre todas las ROI y guarda la info de cada una en results
                    results.append({
                        "case_folder": case_folders[i],
                        "roi_probability": probabilities[i],
                        "roi_label": roi_labels[i],
                        "mammogram_label": mammogram_labels[i],
                    })
    
        # Dataframe con los resultados
        roi_predictions = pd.DataFrame(results)
        print("\nROIs evaluadas:", len(roi_predictions))
        print("Mamografías únicas:", roi_predictions["case_folder"].nunique())
    
        # Agrupar por mamografía: la probabilidad de malignidad de lamamografía es la máxima probabilidadde sus ROI
        mammogram_predictions = (
            roi_predictions.groupby("case_folder") #agrupar por caso
            .agg(
                mammogram_probability=("roi_probability", "max"), #se queda con la probabilidad de malignidad más alta
                mammogram_label=("mammogram_label", "first"), #se queda con el nombre del caso de la primer ROI (todas las ROI tienen el mismo nombre)
                num_rois=("roi_probability", "count"), #número de ROIs de la mamografía analizada
            )
            .reset_index()
        )
    
        # Predicción usando umbral 0.5
        mammogram_predictions["prediction"] = (mammogram_predictions["mammogram_probability"] >= 0.5).astype(int)

        # Arrays
        y_true = mammogram_predictions["mammogram_label"]
        y_pred = mammogram_predictions["prediction"]
        y_prob = mammogram_predictions["mammogram_probability"]

        # Métricas
        metrics = calculate_metrics(y_true, y_pred, y_prob)
    
        # Imprimir resultados
        print("Resultados de la evaluación de DenseNet sobre validación")
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
        output_path = ("../outputs/experiment6_densenet/validation_mammogram_predictions.csv")
        mammogram_predictions.to_csv(output_path, index=False)
        print("\nPredicciones guardadas en:", output_path)