from pathlib import Path
from nerf_model.dataset.lego_dataset import LegoDataset
from nerf_model.config.core import config

def test_lego_dataset_loading():
    lego_dataset = LegoDataset(config.model_config.data_dir)
    assert lego_dataset.images.shape[0] == lego_dataset.poses.shape[0]