import numpy as np
import matplotlib.pyplot as plt
from .base_dataset import BaseDataset

class LegoDataset(BaseDataset):
    """
    Lego dataset consists of 106 images taken of the synthetic Lego bulldozer along with poses and a common focal length value.
    Like the original, we reserve the first 100 images for training and a single test image for validation.
    """

    def __init__(self, root_dir):
        super().__init__(root_dir, "tiny_nerf_data", ".npz")
        self.data = self._load_data()
        self._set_up_data()
        self._sanity_check()

    def _load_data(self):
        return np.load(self.data_path)

    def _set_up_data(self):
        self.images = self.data["images"]
        self.poses = self.data["poses"]
        self.focal_lenght = self.data["focal"]

    def _sanity_check(self):
        assert self.images.shape[0] == self.poses.shape[0], "Number of image is not same as number of poses!"
        print(f"Image shape: {self.images.shape}")
        print(f"Pose shape: {self.poses.shape}")

        self._plot_save_img()

    def _plot_save_img(self):
        img_path = self.root_dir / "test_lego_img.png"
        plt.imshow(self.images[101])
        plt.savefig(img_path)
