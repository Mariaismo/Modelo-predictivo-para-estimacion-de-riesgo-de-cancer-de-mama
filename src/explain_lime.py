### Explicabilidad LIME###

import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from lime import lime_image
from skimage.segmentation import slic
from model import BreastCancerResNet
from config import PROCESSED_DIR

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = (PROJECT_ROOT/ "outputs"/ "experiment5_roi" / "best_model_roi.pth")
CSV_PATH = PROCESSED_DIR / "roi_dataset.csv"
OUTPUT_DIR = (PROJECT_ROOT / "outputs" / "experiment5_roi" / "lime")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

def predict(images):
    """
    LIME proporciona imágenes H x W x 3, pero el modelo trabaja con un único canal.
    Los tres canales contienen la misma imagen, por lo que se utiliza únicamente el primero.
    """

    images = np.asarray(images, dtype=np.float32) #convertir imágenes a un array de NumPy

    if images.ndim == 3:
        images = images[np.newaxis, ...] #añadir una dimensión al array para concocer el nº de imágenes recibidas, se pueden procesar varias a la vez

    # Se queda con el primer canal
    images_gray = images[..., 0]
    tensor = torch.from_numpy(images_gray).unsqueeze(1) #añade la dimensión de ese canal, quedando: (número de imágenes, canales, alto, ancho)
    tensor = tensor.to(device)

    #aplicar el modelo
    with torch.no_grad():
        outputs = model(tensor)
        probabilities = torch.softmax(outputs, dim=1)

    return probabilities.cpu().numpy()

def segmentation_fn(image):
    """
    Divide la imagen en superpíxeles.
    El fondo negro se excluye de la segmentación.
    """

    gray = image[..., 0] #selecciona el primer canal
    foreground_mask = gray > 0 #genera una máscara booleana (True si pixel >0, False en otro caso)
    segments = slic( image, n_segments=500, compactness=10, sigma=1, start_label=1, mask=foreground_mask) #dividir imagen en 500 superpíxeles
    segments[ ~foreground_mask] = 0 #asignar 0 al fondo

    return segments

# Dispositivo
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))

# Modelo
model = BreastCancerResNet().to(device)
state_dict = torch.load(MODEL_PATH, map_location=device)
model.load_state_dict(state_dict)
model.eval()
print("Modelo cargado correctamente.")

# Cargar dataset
df = pd.read_csv(CSV_PATH)
test_df = df[df["split"] == "test"].reset_index(drop=True) #se selecciona solo el conjunto de test
print("ROIs test:", len(test_df))

# Seleccionar caso
idx = 633 # Indice del caso seleccionado para XAI
row = test_df.iloc[idx]
image_path = (PROCESSED_DIR / row["processed_image"]) 
image = np.load(image_path).astype(np.float32) #carga la imagen del caso seleccionado en formato .npy

# Información del caso
print("\nCASO A EXPLICAR")
print("Índice:", idx)
print("Case folder:", row["case_folder"])
print("Patient ID:", row["patient_id"])
print("ROI label:", row["roi_label"])
print("Mammogram label:", row["mammogram_label"])
print("Shape:", image.shape)

# Predicción
image_tensor = (torch.from_numpy(image).unsqueeze(0).unsqueeze(0).to(device)) # modificar dimensiones del tensor a (1(batch),1(canal),512,512)

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

# LIME necesita tres canales: se repite la imagen en RGB
image_rgb = np.stack([ image, image, image], axis=-1)
image_uint8 = np.clip(image_rgb * 255, 0, 255).astype(np.uint8)

# Aplicación de LIME
explainer = lime_image.LimeImageExplainer(random_state=42)
explanation = explainer.explain_instance(image_uint8, predict, top_labels=2,  hide_color=0, num_samples=3000, segmentation_fn=segmentation_fn)
print("LIME calculado.")
label_to_explain = 1 #se explica la clase maligna

# Máscra LIME
lime_image_result, lime_mask = (explanation.get_image_and_mask(label_to_explain, positive_only=False, num_features=10, hide_rest=False)) #máscara con las 10 regiones más importantes

# Máscara para quitar el fondo
background_mask = image == 0 #máscara booleana (True si pixel=0, False en otro caso)
lime_mask = lime_mask.astype(np.float32)
lime_mask[background_mask] = 0 #eliminar fondo

# Mostrar regiones
local_exp = explanation.local_exp[label_to_explain]
print("\nRegiones más relevantes:")
for segment_id, weight in local_exp[:10]:
    print(f"Segmento {segment_id}: "f"{weight:.6f}")

# Visualización
fig, axes = plt.subplots(1, 3, figsize=(18, 6))
# ROI original a la izda
axes[0].imshow(image, cmap="gray")
axes[0].set_title("ROI original")
axes[0].axis("off")
# Regiones LIME en medio
axes[1].imshow(image, cmap="gray")
axes[1].imshow(lime_mask, cmap="RdBu_r", vmin=-1, vmax=1, alpha=0.60) #rojo contribuye a maligno, azul a benigno
axes[1].set_title("LIME - regiones relevantes")
axes[1].axis("off")
# LIME sobre ROI original
axes[2].imshow(image, cmap="gray")
axes[2].imshow(lime_mask, cmap="RdBu_r", vmin=-1, vmax=1, alpha=0.55)
axes[2].set_title("LIME sobre la ROI")
axes[2].axis("off")
#Título
fig.suptitle(f"LIME - Caso {idx} | " f"Real: {row['roi_label']} | " f"P(maligna): {prob_malignant:.3f}", fontsize=14)
fig.tight_layout()

#Guardado
output_path = (OUTPUT_DIR / f"lime_case_{idx}.png")
fig.savefig(output_path, dpi=200, bbox_inches="tight")
print("\nImagen guardada en:", output_path)

plt.show()
plt.close(fig)