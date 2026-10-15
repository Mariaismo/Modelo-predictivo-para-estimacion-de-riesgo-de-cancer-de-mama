### Crea los dataloaders de cada conjunto para los datos de entrada ROI###

from torch.utils.data import DataLoader
from config import PROCESSED_DIR, BATCH_SIZE
from dataset_roi import MamographiesROIDataset


def create_roi_dataloaders():
    train_dataset = MamographiesROIDataset(
        csv_path=PROCESSED_DIR / "roi_dataset.csv",
        image_root=PROCESSED_DIR,
        split="train"
    )

    val_dataset = MamographiesROIDataset(
        csv_path=PROCESSED_DIR / "roi_dataset.csv",
        image_root=PROCESSED_DIR,
        split="val"
    )

    test_dataset = MamographiesROIDataset(
        csv_path=PROCESSED_DIR / "roi_dataset.csv",
        image_root=PROCESSED_DIR,
        split="test"
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True
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