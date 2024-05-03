from nerf_app.dataset.lego_dataset import *
from nerf_app.models.VolumeRendering import VolumeRendering
from nerf_app.utils.calculate_rays import *
from tests import device


def test_volume_integration():
    root_dir = Path("../")
    prep_ds = PrepareData(device=device, root_dir=root_dir)
    train_ds, val_ds = prep_ds.get_data()

    img_h, img_w = val_ds["images"].shape[1:3]
    f_l = val_ds["focal"]
    poses = val_ds["poses"]
    r_o, r_d = calculate_rays(img_h, img_w, f_l, poses[0])

    # Draw stratified samples from example
    rays_o = r_o.view([-1, 3])
    rays_d = r_d.view([-1, 3])
    n_samples = 64
    perturb = True
    inverse_depth = False

    VR = VolumeRendering()

    with torch.no_grad():
        pts, z_vals = VR.stratified_sampling(
            rays_o, rays_d, 2.0, 6.0, n_samples, perturb, inverse_depth
        )
    # Example of input to volume integration
    raw = torch.rand(2500, 64, 4)
    z_vals = z_vals[:2500]
    rays_d = rays_d[:2500]
    print(f"z_vals:{z_vals.shape}")
    print(f"rays_d:{rays_d.shape}")
    # raw: torch.Size([2500, 64, 4])
    # z_vals: torch.Size([2500, 64])
    # rays_d: torch.Size([2500, 3])

    rgb_map, depth_map, acc_map, weights = VR.volume_integration(raw, z_vals, rays_d)
