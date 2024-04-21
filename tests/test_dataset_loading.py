from pathlib import Path

from nerf_app.dataset.lego_dataset import *

def test_dataset():
    root_dir = Path("D:/ML_AI_DL_Projects/projects_repo/nerf")
    prep_ds = PrepareData(root_dir=root_dir)
    train_ds, val_ds = prep_ds.get_data()

    train_dataset = LegoDataset(train_ds)
    validation_dataset = LegoDataset(val_ds)
