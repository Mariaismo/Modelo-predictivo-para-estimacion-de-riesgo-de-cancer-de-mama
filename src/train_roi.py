### Entrenamiento del modelo para ROIs###

import torch
import torch.nn as nn
import pandas as pd
import os
from tqdm import tqdm
from model import BreastCancerResNet
from dataloader_roi import create_roi_dataloaders
from config import EPOCHS

LEARNING_RATE = 5e-5
WEIGHT_DECAY = 1e-4
OUTPUT_DIR = "../outputs/experiment5_roi"

if __name__ == "__main__":
    # Dispositivo
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Dispositivo:", device)

    # Modelo
    model = BreastCancerResNet()
    model = model.to(device)

    # Dataloaders
    train_loader, val_loader, test_loader = (create_roi_dataloaders())

    # Función de pérdida
    criterion = nn.CrossEntropyLoss()

    # Optimizador
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY
    )

    # Información inicial
    print("\nModelo creado:", type(model).__name__)
    print("Train:", len(train_loader.dataset))
    print("Validation:", len(val_loader.dataset))
    print("Test:", len(test_loader.dataset))
    print("Learning rate:", LEARNING_RATE)
    print("Weight decay:", WEIGHT_DECAY)
    print("Batch size:", train_loader.batch_size)
    print("Épocas:", EPOCHS)
    print("Función de pérdida:", type(criterion).__name__)
    print("Optimizador:", type(optimizer).__name__)

    #Generar carpeta de salida
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Variables para guardar historial
    train_losses = []
    val_losses = []
    val_accuracies = []
    best_val_loss = float("inf")
    best_epoch = 0

    #Bucle de entrenamiento
    for epoch in range(EPOCHS):
        print(f"\nÉpoca {epoch + 1}/{EPOCHS}")

        model.train()
        running_loss = 0.0

        for images, roi_labels, _, _ in tqdm(train_loader, desc="Entrenando"):
            images = images.to(device)
            roi_labels = roi_labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, roi_labels)
            loss.backward()
            optimizer.step()
            running_loss += (loss.item()* images.size(0))

        train_loss = (running_loss/ len(train_loader.dataset))
        print(f"Loss de entrenamiento: " f"{train_loss:.4f}")

        # Validación
        model.eval()
        validation_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for images, roi_labels, _, _ in tqdm( val_loader, desc="Validando"):
                images = images.to(device)
                roi_labels = roi_labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, roi_labels)
                validation_loss += (loss.item()  * images.size(0))

                # Predicción
                predictions = outputs.argmax(dim=1)
                correct += (predictions == roi_labels).sum().item()
                total += roi_labels.size(0)

        val_loss = (validation_loss / len(val_loader.dataset))
        val_accuracy = (correct / total)

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        val_accuracies.append(val_accuracy)

        # Guardar mejor modelo
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch + 1
            best_model_path = (f"{OUTPUT_DIR}/best_model_roi.pth")
            torch.save(model.state_dict(), best_model_path)
            print("Nuevo mejor modelo guardado")

        # Mostrar resultados
        print(f"Loss validación: "f"{val_loss:.4f}")
        print(f"Accuracy validación: " f"{val_accuracy:.4f}")

    # Guardar historial
    results = pd.DataFrame({
        "epoch":
            range(1, EPOCHS + 1),
        "train_loss":
            train_losses,
        "val_loss":
            val_losses,
        "val_accuracy":
            val_accuracies
    })

    history_path = (f"{OUTPUT_DIR}/training_history_roi.csv")
    results.to_csv(history_path, index=False)

    # Resumen final
    print("ENTRENAMIENTO ROI FINALIZADO")
    print(f"Mejor época: {best_epoch}")
    print(f"Mejor loss de validación: " f"{best_val_loss:.4f}")
    print("\nArchivos generados:")
    print( f"- {OUTPUT_DIR}/best_model_roi.pth")
    print( f"- {OUTPUT_DIR}/training_history_roi.csv")
