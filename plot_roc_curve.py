###Script para representar la curva ROC###

from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, roc_auc_score


# # Leer el CSV de predicciones del conjunto de test
CSV_PATH = Path(r"G:\TFM\outputs\experiment5_roi\test_mammogram_predictions.csv")
df = pd.read_csv(CSV_PATH)

# Etiquetas reales y probabilidades de malignidad
y_true = df["mammogram_label"].astype(int)
y_score = df["mammogram_probability"].astype(float)

# Calcular curva ROC y AUC
fpr, tpr, thresholds = roc_curve(y_true, y_score)
roc_auc = roc_auc_score(y_true, y_score)
print(f"ROC-AUC: {roc_auc:.4f}")

# Dibujar la curva ROC
plt.figure(figsize=(7, 6))
plt.plot(fpr, tpr, linewidth=2, label=f"ResNet-18 + ROI (AUC = {roc_auc:.4f})")

# Referencia: clasificador aleatorio
plt.plot(
    [0, 1],
    [0, 1],
    linestyle="--",
    linewidth=1.5,
    label="Clasificador aleatorio"
)

plt.xlim(0, 1)
plt.ylim(0, 1.02)

plt.xlabel("Tasa de falsos positivos (1 - especificidad)")
plt.ylabel("Sensibilidad")
plt.title("Curva ROC del modelo en el conjunto de test")

plt.legend(loc="lower right")
plt.grid(alpha=0.2)
plt.tight_layout()

# Guardar la figura en la carpeta de resultados
output_path = CSV_PATH.parent / "roc_curve_test.png"
plt.savefig(output_path, dpi=300, bbox_inches="tight")

plt.show()