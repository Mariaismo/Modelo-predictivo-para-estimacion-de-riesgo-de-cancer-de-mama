### Entrenamiento del modelo DenseNet-121 para ROIs###

import os
import torch
import torch.nn as nn
import pandas as pd
from tqdm import tqdm
from model_densenet import BreastCancerDenseNet
from dataloader_roi import create_roi_dataloaders
from config import EPOCHS

LEARNING_RATE = 5e-5
WEIGHT_DECAY = 1e-4
OUTPUT_DIR = "../outputs/experiment6_densenet"

if __name__ == "__main__":
    # Dispositivo
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Dispositivo:", device)

    # Modelo
    model = BreastCancerDenseNet()
    model = model.to(device)

    # Dataloaders
    train_loader, val_loader, test_loader = (create_roi_dataloaders())

    # Loss
    criterion = nn.CrossEntropyLoss()

    # Optimizador
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY
    )

    # Crear carpeta de salida
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("\nModelo creado:", type(model).__name__)
    print("Train:", len(train_loader.dataset))
    print("Validation:", len(val_loader.dataset))
    print("Test:", len(test_loader.dataset))
    print("Learning rate:", LEARNING_RATE)
    print("Weight decay:", WEIGHT_DECAY)
    print("Batch size:", train_loader.batch_size)
    print("Épocas de prueba:", EPOCHS)
    print("Función de pérdida:", type(criterion).__name__)
    print("Optimizador:", type(optimizer).__name__)

    train_losses = []
    val_losses = []
    val_accuracies = []
    best_val_loss = float("inf")
    best_epoch = 0

    # Bucle de entrenamiento
    for epoch in range(EPOCHS):
        print(f"\nÉpoca {epoch + 1}/{EPOCHS}")

        model.train()
        running_loss = 0.0
        for images, roi_labels, _, _ in tqdm(train_loader, desc="Entrenando"):
            images = images.to(device, non_blocking=True)
            roi_labels = roi_labels.to(device, non_blocking=True)
            optimizer.zero_grad()
            outputs = model( images)
            loss = criterion(outputs, roi_labels)

            loss.backward()
            optimizer.step()
            running_loss += (
                loss.item()
                * images.size(0)
            )

        train_loss = (running_loss / len(train_loader.dataset))
        print(f"Loss de entrenamiento: " f"{train_loss:.4f}")

        # Validation
        model.eval()

        validation_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for images, roi_labels, _, _ in tqdm( val_loader, desc="Validando"):

                images = images.to(device, non_blocking=True)
                roi_labels = roi_labels.to(device, non_blocking=True)
                outputs = model(images)

                loss = criterion(outputs, roi_labels)
                validation_loss += (loss.item()* images.size(0))

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
            torch.save(model.state_dict(),f"{OUTPUT_DIR}/best_model_densenet.pth")
            print("Nuevo mejor modelo guardado")

        print(f"Loss validación: " f"{val_loss:.4f}")
        print(f"Accuracy validación: " f"{val_accuracy:.4f}")

    results = pd.DataFrame({
        "epoch":range(1, EPOCHS + 1),
        "train_loss":train_losses,
        "val_loss":val_losses,
        "val_accuracy":val_accuracies
    })

    history_path = (f"{OUTPUT_DIR}/training_history_densenet.csv")
    results.to_csv(history_path, index=False)

    # Resumen
    print("ENTRENAMIENTO DENSENET FINALIZADO")
    print(f"Mejor época: {best_epoch}")
    print(f"Mejor loss de validación: " f"{best_val_loss:.4f}")
    print("\nArchivos generados:")
    print( f"- {OUTPUT_DIR}/best_model_densenet.pth")
    print( f"- {OUTPUT_DIR}/training_history_densenet.csv")

