import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

import matplotlib.pyplot as plt

from nerf_model.dataset.lego_dataset import LegoDataset
from nerf_model.config.core import config
from nerf_model.tools.calculate_rays import get_rays
from nerf_model.models.VolumeSampling import VolumeSampling

from nerf_model.models.PositionalEncoder import PositionalEncoder

def test_positional_encoder():
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

    print("Input Points")
    print(pts.shape)
    print("")
    print("Distances Along Ray")
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
    
    #Test Positional Encoder
    encoder = PositionalEncoder(3,10)
    view_encoder = PositionalEncoder(3,4)
    
    pts_flatten = pts.reshape(-1,3)
    print(f"pts_flatten shape:{pts_flatten.shape}")

    view_dir = rays_d / torch.norm(rays_d, dim=-1, keepdim=True)
    print(f"View dir shape:{view_dir.shape}")
    flatten_view_dir = view_dir[:, None, ...].expand(pts.shape).reshape((-1, 3))
    print(f"flatten_view_dir shape:{flatten_view_dir.shape}")
    
    #Encode data
    encoded_points = encoder(pts_flatten)
    encoded_view = view_encoder(flatten_view_dir)
    
    print('Encoded Points')
    print(encoded_points.shape)
    print(torch.min(encoded_points), torch.max(encoded_points), torch.mean(encoded_points))
    print('')

    print('Encoded Viewdirs')
    print(encoded_view.shape)
    print(torch.min(encoded_view), torch.max(encoded_view), torch.mean(encoded_view))
    print('')
