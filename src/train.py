### Entrenamiento del modelo para mamografías completas###

import torch
import torch.nn as nn
import pandas as pd
from model import BreastCancerResNet
from dataloader import create_dataloaders
from tqdm import tqdm
from config import EPOCHS

#Configurar el dispositivo para el entrenamiento: cuda si hay GPU, CPU en caso contrario
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Dispositivo:", device)

# Modelo
model = BreastCancerResNet()
model = model.to(device) #mueve los pesos al dispositivo seleccionado
train_loader, val_loader, test_loader = create_dataloaders()

criterion = nn.CrossEntropyLoss() #función de pérdida para comparar lo predicho con la etiqueta real

optimizer = torch.optim.Adam( #optimizador, va modificando los pesos en función del resultado d ela función de pérdida
    model.parameters(),
    lr=0.00005,
    weight_decay=0.0001
)

print("Modelo creado:", type(model).__name__)
print("Train:", len(train_loader.dataset))
print("Validation:", len(val_loader.dataset))
print("Test:", len(test_loader.dataset))
print("Función de pérdida:", type(criterion).__name__)
print("Optimizador:", type(optimizer).__name__)

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

    for images, labels in tqdm(train_loader, desc="Entrenando"):
        images = images.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * images.size(0)

    train_loss = running_loss / len(train_loader.dataset)
    print(f"Loss de entrenamiento: {train_loss:.4f}")

    #Validación
    model.eval() #el modelo está en validación
    validation_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in tqdm(val_loader, desc="Validando"):
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            validation_loss += loss.item() * images.size(0)
            predictions = outputs.argmax(dim=1) #elige la clase con mayor probabilidad
            correct += (predictions == labels).sum().item() #cuenta cuántas predicciones son correctas
            total += labels.size(0)

    val_loss = validation_loss / len(val_loader.dataset)
    val_accuracy = correct / total

    train_losses.append(train_loss)
    val_losses.append(val_loss)
    val_accuracies.append(val_accuracy)

    if val_loss < best_val_loss: #guardar el modelo con mejores resultados
        best_val_loss = val_loss
        best_epoch = epoch + 1
        torch.save(
            model.state_dict(),
            "best_model.pth"
        )
        print("Nuevo mejor modelo guardado")

    print(f"Loss validación: {val_loss:.4f}")
    print(f"Accuracy validación: {val_accuracy:.4f}")

results = pd.DataFrame({
    "epoch": range(1, EPOCHS + 1),
    "train_loss": train_losses,
    "val_loss": val_losses,
    "val_accuracy": val_accuracies
})

results.to_csv(
    "training_history.csv",
    index=False
)

#Resumen final
print("ENTRENAMIENTO FINALIZADO")
print(f"Mejor época: {best_epoch}")
print(f"Mejor loss de validación: {best_val_loss:.4f}")
print("\nArchivos generados:")
print("- best_model.pth")
print("- training_history.csv")