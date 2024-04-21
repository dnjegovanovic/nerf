from pathlib import Path

import matplotlib.pyplot as plt
import torch
from mpl_toolkits.mplot3d import axes3d

from nerf_app.dataset.lego_dataset import *

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
if __name__ == "__main__":
    root_dir = Path("D:/ML_AI_DL_Projects/projects_repo/nerf")
    prep_ds = PrepareData(root_dir=root_dir)
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
    plt.savefig("../../output/visualize_camera_position_2.png")
