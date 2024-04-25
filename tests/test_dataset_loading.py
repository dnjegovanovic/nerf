from nerf_app.dataset.lego_dataset import *
from tests import device


def test_dataset():
    root_dir = Path("D:/ML_AI_DL_Projects/projects_repo/nerf")
    prep_ds = PrepareData(device=device, root_dir=root_dir)
    train_ds, val_ds = prep_ds.get_data()

    LegoDataset(train_ds)
    LegoDataset(val_ds)
