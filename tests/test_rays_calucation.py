from nerf_app.dataset.lego_dataset import *
from nerf_app.utils.calculate_rays import *
from tests import device


def test_rays_calc():
    root_dir = Path("D:/ML_AI_DL_Projects/projects_repo/nerf")
    prep_ds = PrepareData(device=device, root_dir=root_dir)
    train_ds, val_ds = prep_ds.get_data()

    img_h, img_w = val_ds["images"].shape[1:3]
    f_l = val_ds["focal"]
    poses = val_ds["poses"]
    r_o, r_d = calculate_rays(img_h, img_w, f_l, poses[0])

    assert r_o.shape == r_d.shape
