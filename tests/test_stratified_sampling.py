import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

import matplotlib.pyplot as plt

from nerf_model.dataset.lego_dataset import LegoDataset
from nerf_model.config.core import config
from nerf_model.tools.calculate_rays import get_rays
from nerf_model.models.VolumeSampling import VolumeSampling


def test_stratified_sampling():
    lego_dataset = LegoDataset(config.model_config.data_dir)
    # Gather as torch tensors
    images = lego_dataset.images[: config.app_config.training_sample].to(device)
    focal = lego_dataset.focal_length.to(device)
    testpose = lego_dataset.poses[config.app_config.testimg_idx].to(device)
    # Grab rays from sample image
    height, width = images.shape[1:3]
    with torch.no_grad():
        ray_origin, ray_direction = get_rays(height, width, focal, testpose)

    print("Ray Origin")
    print(ray_origin.shape)
    print(ray_origin[height // 2, width // 2, :])
    print("")

    print("Ray Direction")
    print(ray_direction.shape)
    print(ray_direction[height // 2, width // 2, :])
    print("")

    rays_o = ray_origin.view([-1, 3])
    rays_d = ray_direction.view([-1, 3])

    volume_sampling = VolumeSampling(
        config.stratified_config.strf_samp_option["n_samples"],
        config.stratified_config.strf_samp_option["perturb"],
        config.stratified_config.strf_samp_option["inverse_depth"],
    )

    with torch.no_grad():
        pts, z_vals = volume_sampling.stratified_sampling(
            rays_o,
            rays_d,
            config.stratified_config.strf_samp_option["near"],
            config.stratified_config.strf_samp_option["far"],
        )

    print('Input Points')
    print(pts.shape)
    print('')
    print('Distances Along Ray')
    print(z_vals.shape)

    y_vals = torch.zeros_like(z_vals)
    volume_sampling.perturb = False
    _, z_vals_unperturbed = volume_sampling.stratified_sampling(
        rays_o,
        rays_d,
        config.stratified_config.strf_samp_option["near"],
        config.stratified_config.strf_samp_option["far"],
    )
    plt.plot(z_vals_unperturbed[0].cpu().numpy(), 1 + y_vals[0].cpu().numpy(), "b-o")
    plt.plot(z_vals[0].cpu().numpy(), y_vals[0].cpu().numpy(), "r-o")
    plt.ylim([-1, 2])
    plt.title("Stratified Sampling (blue) with Perturbation (red)")
    ax = plt.gca()
    ax.axes.yaxis.set_visible(False)
    plt.grid(True)
    img_path = config.model_config.data_dir + "/stratified_sampling.png"
    plt.savefig(img_path)
    plt.close()
