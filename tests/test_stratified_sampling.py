from nerf_app.dataset.lego_dataset import *
from nerf_app.models.VolumeRendering import VolumeRendering
from nerf_app.utils.calculate_rays import *
from tests import device


def test_stratified_sampling():
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

    print("Input Points")
    print(pts.shape)
    print("-" * 80)
    print("Distances Along Ray")
    print(z_vals.shape)

    assert pts.shape[0] == z_vals.shape[0]
    assert pts.shape[1] == z_vals.shape[1]
