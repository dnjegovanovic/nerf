from pathlib import Path

from nerf_app.dataset.lego_dataset import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def test_dataset():
    root_dir = Path("D:/ML_AI_DL_Projects/projects_repo/nerf")
    prep_ds = PrepareData(device=device, root_dir=root_dir)
    train_ds, val_ds = prep_ds.get_data()

    train_dataset = LegoDataset(train_ds)
    validation_dataset = LegoDataset(val_ds)
