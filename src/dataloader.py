### Crea los dataloaders de cada conjunto para los datos de entrada mamografías completas###

from torch.utils.data import DataLoader
from config import PROCESSED_DIR, BATCH_SIZE
from dataset import MamographiesDataset


def create_dataloaders():

    train_dataset = MamographiesDataset(
        csv_path=PROCESSED_DIR / "dataset.csv",
        image_root=PROCESSED_DIR,
        split="train"
    )

    val_dataset = MamographiesDataset(
        csv_path=PROCESSED_DIR / "dataset.csv",
        image_root=PROCESSED_DIR,
        split="val"
    )

    test_dataset = MamographiesDataset(
        csv_path=PROCESSED_DIR / "dataset.csv",
        image_root=PROCESSED_DIR,
        split="test"
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True #en el entrenamiento interesa mezclar las imágenes
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False
    )

    return train_loader, val_loader, test_loader
