### Carga las mamografías completas procesadas, aplica augmentation en entrenamiento y devuelve la imagen y su etiqueta###

import pandas as pd
import numpy as np
import torch
from pathlib import Path
from torch.utils.data import Dataset
from torchvision import transforms


class MamographiesDataset(Dataset):
    
    def __init__(self, csv_path, image_root, split):
        self.data = pd.read_csv(csv_path)
        self.data = self.data[self.data["split"] == split].reset_index(drop=True)
        self.image_root = Path(image_root)
        self.split = split
        # Transformaciones para entrenamiento
        if split == "train":
            self.transform = transforms.Compose([
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomRotation(degrees=10),
            ])
        # Sin augmentation para validación y test
        else:
            self.transform = None

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx] #obtener fila de la mamografía de la iteración actual
        image_path = self.image_root / row["processed_image"] #ruta a la imagen .npy
        image = np.load(image_path)
        image = torch.from_numpy(image)#convertir imagen en tensor
        image = image.unsqueeze(0) #añadir canal a la imagen (1x512x512) escala de grises
        # Aplicar augmentation únicamente en train
        if self.transform is not None:
            image = self.transform(image)
        label = torch.tensor(row["label"], dtype=torch.long) #convertir en tensor la etiqueta 0/1
        return image, label

