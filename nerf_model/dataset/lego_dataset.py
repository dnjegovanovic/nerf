import numpy as np
import torch
import matplotlib.pyplot as plt
from .base_dataset import BaseDataset
from nerf_model.tools.calculate_rays import *


class LegoDataset(BaseDataset):
    """
    Lego dataset consists of 106 images taken of the synthetic Lego bulldozer along with poses and a common focal length value.
    Like the original, we reserve the first 100 images for training and a single test image for validation.
    """

    def __init__(self, root_dir: str, debug=False):
        super().__init__(root_dir, "tiny_nerf_data", ".npz")
        self.data = self._load_data()
        self._set_up_data()
        self._sanity_check()
        if debug:
            self._plot_camera_position()

    def _load_data(self):
        return np.load(self.data_path)

    def _set_up_data(self):
        self.images = torch.from_numpy(self.data["images"])
        self.poses = torch.from_numpy(self.data["poses"])
        self.focal_length = torch.from_numpy(self.data["focal"])

        self.img_height, self.img_width = self.images[1:3]

    def get_all_rays(self, number):
        all_rays = [
            [get_rays(self.img_height, self.img_width, self.focal_length, p)]
            for p in self.poses
        ]

        return all_rays

    def _sanity_check(self):
        assert (
            self.images.shape[0] == self.poses.shape[0]
        ), "Number of image is not same as number of poses!"
        print(f"Image shape: {self.images.shape}")
        print(f"Pose shape: {self.poses.shape}")

        self._plot_save_img()

    def _plot_save_img(self):
        img_path = self.root_dir / "test_lego_img.png"
        plt.imshow(self.images[101].cpu().detach().numpy())
        plt.savefig(img_path)
        plt.close()

    def _plot_camera_position(self):
        directions = np.stack(
            [
                np.sum([0, 0, -1] * pose[:3, :3].cpu().detach().numpy(), axis=-1)
                for pose in self.poses
            ]
        )
        origins = self.poses[:, :3, -1].cpu().detach().numpy()
        print(f"Origins:{origins[0]}")

        ax = plt.figure(figsize=(12, 8)).add_subplot(projection="3d")
        _ = ax.quiver(
            origins[..., 0].flatten(),
            origins[..., 1].flatten(),
            origins[..., 2].flatten(),
            directions[..., 0].flatten(),
            directions[..., 1].flatten(),
            directions[..., 2].flatten(),
            length=0.5,
            normalize=True,
        )
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("z")
        # plt.show()
        img_path = self.root_dir / "camera_vis.png"
        plt.savefig(img_path)
        plt.close()
