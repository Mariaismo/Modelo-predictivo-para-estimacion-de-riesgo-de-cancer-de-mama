### Carga las ROI, aplica augmentation en entrenamiento y devuelve la imagen, su etiqueta y la información de la mamografía asociada###

import pandas as pd
import numpy as np
import torch
from pathlib import Path
from torch.utils.data import Dataset
from torchvision import transforms


class MamographiesROIDataset(Dataset):

    def __init__(self, csv_path, image_root, split):
        self.data = pd.read_csv(csv_path, sep=";")
        self.data = (self.data[self.data["split"] == split].reset_index(drop=True))
        self.image_root = Path(image_root)
        self.split = split

        # Augmentation solo para entrenamiento
        if split == "train":
            self.transform = transforms.Compose([
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomRotation(degrees=10),
            ])
        else:
            self.transform = None


    def __len__(self):
        return len(self.data)


    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        image_path = (self.image_root /row["processed_image"])
        image = np.load(image_path)
        image = torch.from_numpy(image).float()

        # 1 canal
        image = image.unsqueeze(0)

        # Augmentation solo train
        if self.transform is not None:
            image = self.transform(image)

        # Etiqueta de esta ROI
        roi_label = torch.tensor(row["roi_label"], dtype=torch.long)

        # Información para poder agrupar posteriormente por mamografía
        case_folder = row["case_folder"]

        mammogram_label = torch.tensor(row["mammogram_label"], dtype=torch.long)

        return (image, roi_label, case_folder, mammogram_label)