from pathlib import Path, PureWindowsPath

import numpy as np
import torch
from torch.utils.data import Dataset


class PrepareData:
    def __init__(self, device, root_dir: Path, validation_split: float = 0.1):
        self.validation_split = validation_split
        self.lego_root_dir = Path(PureWindowsPath(root_dir))
        self.data_path = self.lego_root_dir / "data/tiny_nerf_data.npz"
        if self.data_path.exists():
            self.data = np.load(self.data_path)
        else:
            raise FileNotFoundError(self.data_path)

        self.images = torch.from_numpy(self.data["images"]).to(device)
        self.poses = torch.from_numpy(self.data["poses"]).to(device)
        self.focal_length = torch.from_numpy(self.data["focal"]).to(device)

    def get_data(self):
        self.indices = list(range(len(self.images)))
        print("-" * 80)
        print(f"Start idx: {self.indices[:10]}")
        np.random.seed(42)
        np.random.shuffle(self.indices)
        print(f"Start idx shuffle: {self.indices[:10]}")
        print("-" * 80)
        split = int(np.floor((1.0 - self.validation_split) * len(self.images)))
        train_img_ds, val_img_ds = self.images[:split], self.images[split:]
        train_pos_ds, val_pos_ds = self.poses[:split], self.poses[split:]
        return {
            "images": train_img_ds,
            "poses": train_pos_ds,
            "focal": self.focal_length,
        }, {"images": val_img_ds, "poses": val_pos_ds, "focal": self.focal_length}


class LegoDataset(Dataset):
    def __init__(self, data):
        self.data = data
        self._display_dataset_property()
        self._sanity_check()

    def _display_dataset_property(self):
        print("-" * 80)
        print(f'Images shape: {self.data["images"].shape}')
        print(f'Poses shape: {self.data["poses"].shape}')
        print(f'Focal length: {self.data["focal"]}')

    def _sanity_check(self):
        assert (
            self.data["images"].shape[0] == self.data["poses"].shape[0]
        ), "Number of images is not same as number of poses"

    def __len__(self):
        return len(self.data["images"])

    def __getitem__(self, idx):
        batch_data = {
            "images": self.data["images"][idx],
            "poses": self.data["poses"][idx],
            "focal": self.data["focal"],
        }

        return batch_data
