### Explicabilidad SHAP###

import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shap
from pathlib import Path
from scipy.ndimage import gaussian_filter
from model import BreastCancerResNet
from config import PROCESSED_DIR

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = (PROJECT_ROOT / "outputs" / "experiment5_roi" / "best_model_roi.pth")
CSV_PATH = PROCESSED_DIR / "roi_dataset.csv"
OUTPUT_DIR = (PROJECT_ROOT / "outputs" / "experiment5_roi" / "shap")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

#Dispositivo
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)

#Modelo
model = BreastCancerResNet().to(device)
state_dict = torch.load(MODEL_PATH, map_location=device)
model.load_state_dict(state_dict)
model.eval()

# Cargar dataset de test
df = pd.read_csv(CSV_PATH)
test_df = df[df["split"] == "test"].reset_index(drop=True)

# Seleccionar caso
idx = 276 # Indice del caso seleccionado para XAI
row = test_df.iloc[idx]
image_path = (PROCESSED_DIR / row["processed_image"])
image = np.load(image_path).astype(np.float32)

# Información del caso
print("CASO A EXPLICAR")
print("Índice:", idx)
print("Case folder:", row["case_folder"])
print("ROI label:", row["roi_label"])
print("Mammogram label:", row["mammogram_label"])
print("Shape:", image.shape)

# Predicción
image_tensor = (torch.from_numpy(image).unsqueeze(0).unsqueeze(0).to(device))

with torch.no_grad():
    output = model(image_tensor) #aplicar el modelo
    probabilities = torch.softmax(output, dim=1) #probabilidades para las dos clases

prob_benign = probabilities[0, 0].item()
prob_malignant = probabilities[0, 1].item()
prediction = int(prob_malignant >= 0.5) #si la probabilidad de maligno es >0.5, entonces es maligna

print("\nPredicción:")
print(f"Probabilidad benigna: {prob_benign:.4f}")
print(f"Probabilidad maligna: {prob_malignant:.4f}")
print(f"Predicción: {prediction}")
print(f"Etiqueta real: {row['roi_label']}")

# Utilizamos únicamente imágenes del conjunto de entrenamiento.
# El conjunto de test NO se utiliza como background.

# Seleccionar 10 ROIs benignas y 10 malignas del conjunto de test para usar de referencia
train_df = df[df["split"] == "train"].copy()
train_benign = train_df[train_df["roi_label"] == 0].sample(n=10, random_state=42)
train_malignant = train_df[train_df["roi_label"] == 1].sample(n=10, random_state=42)
background_df = pd.concat([train_benign, train_malignant]).reset_index(drop=True) #crear un dataframe con esas referencias

# Recorrer dataframe de referencia y guardar imágenes en una lista
background_images = []
for _, background_row in background_df.iterrows():
    background_path = (PROCESSED_DIR / background_row["processed_image"])
    background_image = np.load(background_path).astype(np.float32)
    background_images.append(background_image)

# Convertir imágenes a tensores
background = np.stack(background_images, axis=0)
background_tensor = (torch.from_numpy(background).unsqueeze(1).to(device))

# Aplicación de SHAP
explainer = shap.GradientExplainer(model, background_tensor)
shap_values = explainer.shap_values(image_tensor)
print("SHAP calculado")

values = np.array(shap_values) #convertir imágenes a NumPy

if values.ndim == 5: #las dimensiones esperadas son 5 (nº de imágenes, canal, altura, anchura, clase)
    values = values[0] #desaparece la primera dimensión (canal, altura, anchura, clase)
    values = values[0] #desaparece la segunda dimensión (altura, anchura, clase)
    malignant_map = values[..., 1] #seleccionar la clase maligna
elif values.ndim == 4: #(canal, altura, anchura, clase) o (nº de imágenes, canal, altura, anchura)
    values = values[0] #desaparece la primera dimensión (altura, anchura, clase) o (canal, altura, anchura)
    if values.shape[-1] == 2: # si (altura, anchura, clase)
        malignant_map = values[..., 1] #seleccionar la clase maligna
    else:
        malignant_map = values[0] #desaparece la primera dimensión (altura, anchura)
else:
    raise ValueError(f"Forma SHAP inesperada: {values.shape}")

print("Mapa SHAP malignidad:", malignant_map.shape)

# Visualización
#Suavizado
malignant_map_smooth = gaussian_filter(malignant_map, sigma=1)
print("Mapa SHAP suavizado:", malignant_map_smooth.shape)
#Normalizado
max_abs = np.percentile(np.abs(malignant_map_smooth), 99) #se calcula el percentil 99 para eliminar extremos
if max_abs == 0:
    max_abs = 1.0
shap_display = np.clip(malignant_map_smooth, -max_abs, max_abs) #los valores dentro de los límites se mantienen, los exteriores se truncan

fig, axes = plt.subplots(1, 3, figsize=(18, 6))
#ROI original a la izda
axes[0].imshow(image, cmap="gray")
axes[0].set_title("ROI original")
axes[0].axis("off")
#Mapa SHAP en medio
im = axes[1].imshow(shap_display, cmap="RdBu_r", vmin=-max_abs, vmax=max_abs) #rojo contribuye a maligno, azul a benigno
axes[1].set_title("SHAP - contribución a malignidad")
axes[1].axis("off")
fig.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04)
#SHAP sobre ROI a la dcha
axes[2].imshow(image, cmap="gray")
axes[2].imshow(shap_display, cmap="RdBu_r", vmin=-max_abs, vmax=max_abs, alpha=0.55)
axes[2].set_title("SHAP sobre la ROI")
axes[2].axis("off")
fig.suptitle( f"SHAP - Caso {idx} | " f"Real: {row['roi_label']} | " f"P(maligna): {prob_malignant:.3f}", fontsize=14)
plt.tight_layout()

# Guardar
output_path = (OUTPUT_DIR / f"shap_case_{idx}_gradient_smooth.png")
plt.savefig(output_path, dpi=200, bbox_inches="tight")
plt.show()
print("\nImagen guardada en:", output_path)