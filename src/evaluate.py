### Evaluación del conjunto de test para mamografías completas###

import torch
from model import BreastCancerResNet
from dataloader import create_dataloaders
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix

if __name__ == "__main__":
    # Dispositivo
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Dispositivo:", device)

    #Datos para test
    _, _, test_loader = create_dataloaders()
    print("Test:", len(test_loader.dataset))

    # Modelo
    model = BreastCancerResNet()
    model.load_state_dict(torch.load("best_model.pth", map_location=device)) #cargar el mejor del entrenamiento
    model = model.to(device)
    model.eval()

    # Inferencia
    all_labels = []
    all_predictions = []
    all_probabilities = []

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            outputs = model(images)
            probabilities = torch.softmax(outputs, dim=1) #devuelve la probabilidad de pertenencia a cada clase
            predictions = outputs.argmax(dim=1) #devuelve la clase (0/1) con mayor probabilidad
            all_labels.extend(labels.numpy()) #diagnóstico real
            all_predictions.extend(predictions.cpu().numpy()) #diagnóstico del modelo
            all_probabilities.extend(probabilities[:, 1].cpu().numpy()) #probabilidad de que sea maligno

    #Resultados
    print("\nEvaluación completada")
    print("Número de muestras:", len(all_labels))
    print("Primeras etiquetas reales:", all_labels[:10])
    print("Primeras predicciones:", all_predictions[:10])
    print("Primeras probabilidades:", all_probabilities[:10])

    #Métricas
    accuracy = accuracy_score(all_labels, all_predictions)
    precision = precision_score(all_labels, all_predictions)
    sensitivity = recall_score(all_labels, all_predictions)
    f1 = f1_score(all_labels, all_predictions)
    roc_auc = roc_auc_score(all_labels, all_probabilities)
    cm = confusion_matrix(all_labels, all_predictions)
    tn, fp, fn, tp = cm.ravel()
    specificity = tn / (tn + fp)

    print("Resultados de la evaluación sobre test")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Sensibilidad: {sensitivity:.4f}")
    print(f"Especificidad: {specificity:.4f}")
    print(f"F1-score: {f1:.4f}")
    print(f"ROC-AUC: {roc_auc:.4f}")
    print("\nMatriz de confusión:")
    print(cm)
    print("\nTN:", tn)
    print("FP:", fp)
    print("FN:", fn)
    print("TP:", tp)
