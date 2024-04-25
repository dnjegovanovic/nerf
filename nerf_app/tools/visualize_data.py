from pathlib import Path

import matplotlib.pyplot as plt
import seaborn as sns
import torch

from nerf_app.dataset.lego_dataset import *
from nerf_app.models.PositionalEncoder import PositionalEncoder
from nerf_app.models.VolumeRendering import VolumeRendering
from nerf_app.utils.calculate_rays import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def visualize_cameras():
    """
    Recall that NeRF processes inputs from a field of positions (x,y,z) and view directions (θ,φ). To gather these input points,
    we need to apply inverse rendering to the input images. More concretely,
    we draw projection lines through each pixel and across the 3D space, from which we can draw samples.

    To sample points from the 3D space beyond our image,
    we first start from the initial pose of every camera taken in the photo set.
    With some vector math, we can convert these 4x4 pose matrices into a 3D coordinate denoting the origin
    and a 3D vector indicating the direction. The two together describe a vector that indicates where a camera was pointing when the photo was taken.

    The code in the cell below illustrates this by drawing arrows that depict the origin and the direction of every frame.
    """

    root_dir = Path("D:/ML_AI_DL_Projects/projects_repo/nerf")
    prep_ds = PrepareData(device, root_dir=root_dir)
    train_ds, val_ds = prep_ds.get_data()
    # Take direction vector from pose file
    # For eatch pose take 3x3 matrix(without translation vector) multiplay by -1 z axis to pint to the center
    dirs = np.stack(
        [
            np.sum([0, 0, -1] * pose[:3, :3].cpu().detach().numpy(), axis=-1)
            for pose in train_ds["poses"]
        ]
    )

    # Take origin
    origins = train_ds["poses"][:, :3, -1]

    # PLot data
    ax = plt.figure(figsize=(12, 8)).add_subplot(projection="3d")
    _ = ax.quiver(
        origins[..., 0].flatten(),
        origins[..., 1].flatten(),
        origins[..., 2].flatten(),
        dirs[..., 0].flatten(),
        dirs[..., 1].flatten(),
        dirs[..., 2].flatten(),
        length=0.5,
        normalize=True,
    )
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("z")
    plt.savefig("../../output/images/visualize_camera_position_2.png")


def visualize_calucated_rays():
    """
    Visualize projection from origin to image
    You will be able to se how image grid lokk from one image plane
    """
    root_dir = Path("D:/ML_AI_DL_Projects/projects_repo/nerf")
    prep_ds = PrepareData(device=device, root_dir=root_dir)
    train_ds, val_ds = prep_ds.get_data()

    img_h, img_w = val_ds["images"].shape[1:3]
    f_l = val_ds["focal"]
    poses = val_ds["poses"]
    r_o, r_d = calculate_rays(img_h, img_w, f_l, poses[1])
    print(f"Ray Origin shape:{r_o.shape}")
    print(f"Ray Direc shape:{r_d.shape}")
    # Take origin
    rays_o = r_o.view([-1, 3])
    rays_d = r_d.view([-1, 3])
    print(f"Ray Origin shape:{rays_o.shape}")
    print(f"Ray Direc shape:{rays_d.shape}")

    scale_fac = 6
    origins = rays_o[:1000].cpu().detach().numpy() * scale_fac
    dirs = rays_d[:1000].cpu().detach().numpy() * scale_fac
    # PLot data
    ax = plt.figure(figsize=(20, 20)).add_subplot(projection="3d")
    _ = ax.quiver(
        origins[..., 0].flatten(),
        origins[..., 1].flatten(),
        origins[..., 2].flatten(),
        dirs[..., 0].flatten(),
        dirs[..., 1].flatten(),
        dirs[..., 2].flatten(),
        length=1.0,
        normalize=True,
    )
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("z")
    plt.savefig("../../output/images/visualize_calcul_rays.png")


def visualize_positional_encoder_test():
    encoder = PositionalEncoder(3, 5)
    input_tensorf = torch.rand((10000, 3))
    print(f"input_tensorf: {input_tensorf.shape}")
    plt.figure()
    sns.heatmap(input_tensorf, cmap="GnBu")
    plt.savefig("../../output/images/input_tensorf_positiona_encoding.png")

    encoder_rez = encoder(input_tensorf)

    sns.heatmap(encoder_rez, cmap="GnBu")
    plt.savefig("../../output/images/encoded_positiona_encoding.png")

    print("Encoded Points")
    print(encoder_rez.shape)
    print(torch.min(encoder_rez), torch.max(encoder_rez), torch.mean(encoder_rez))
    print("-" * 80)


def visualize_stratified_sampling():
    root_dir = Path("D:/ML_AI_DL_Projects/projects_repo/nerf")
    prep_ds = PrepareData(device=device, root_dir=root_dir)
    train_ds, val_ds = prep_ds.get_data()

    img_h, img_w = val_ds["images"].shape[1:3]
    f_l = val_ds["focal"]
    poses = val_ds["poses"]
    r_o, r_d = calculate_rays(img_h, img_w, f_l, poses[0])

    # Draw stratified samples from example
    rays_o = r_o.view([-1, 3])
    rays_d = r_d.view([-1, 3])
    n_samples = 8
    perturb = True
    inverse_depth = False

    strat_sampling = VolumeRendering()

    with torch.no_grad():
        pts, z_vals = strat_sampling.stratified_sampling(
            rays_o, rays_d, 2.0, 6.0, n_samples, perturb, inverse_depth
        )
        _, z_vals_unperturbed = strat_sampling.stratified_sampling(
            rays_o,
            rays_d,
            2.0,
            6.0,
            n_samples,
            perturb=False,
            inverse_depth=inverse_depth,
        )

    print("Input Points")
    print(pts.shape)
    print("-" * 80)
    print("Distances Along Ray")
    print(z_vals.shape)

    y_vals = torch.zeros_like(z_vals)
    plt.plot(z_vals[0].cpu().numpy(), 1 + y_vals[0].cpu().numpy(), "b-o")
    plt.plot(z_vals_unperturbed[0].cpu().numpy(), y_vals[0].cpu().numpy(), "r-o")
    plt.ylim([-1, 2])
    plt.title("Stratified Sampling (blue) with Perturbation (red) without Perturb")
    ax = plt.gca()
    ax.axes.yaxis.set_visible(False)
    plt.savefig("../../output/images/stratified_sampling_01.png")


if __name__ == "__main__":
    # visualize_cameras()
    # visualize_calucated_rays()
    # visualize_positional_encoder_test()
    visualize_stratified_sampling()
