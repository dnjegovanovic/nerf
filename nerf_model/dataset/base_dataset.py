from torch.utils.data import Dataset
from pathlib import Path

from torch.utils.data.dataset import Dataset


class BaseDataset(Dataset):
    def __init__(self, root_dir, base_name, extension):
        self.dataset_id = hash(str(root_dir) + "_" + extension)
        self.root_dir = Path(root_dir)
        self.base_name = base_name
        self.extension = extension

        self.data_path = root_dir + base_name + extension

    def __getitem__(self, index):
        pass

    def __len__(self):
        pass


class SplicedDataset(Dataset):
    def __init__(self, images, poses):
        assert images.shape[0] > 0
        assert poses.shape[0] > 0

        self.images = images
        self.poses = poses
        self.lenght = self.images.shape[0]

    def __len__(self):
        return self.lenght

    def __getitem__(self, index):
        return [self.images[index], self.poses[index]]


class SplicedRays(Dataset):
    def __init__(self, rays_o, rays_d, images):
        assert rays_o.shape[0] > 0
        assert rays_d.shape[0] > 0

        self.rays_o = rays_o
        self.rays_d = rays_d
        self.images = images
        self.lenght = self.images.shape[0]

    def __len__(self):
        return self.lenght

    def __getitem__(self, index):
        return {
            "rays_o": self.rays_o[index],
            "rays_d": self.rays_d[index],
            "images": self.images,
        }
