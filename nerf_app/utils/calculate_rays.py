from typing import Tuple

import torch

"""
Use pin-hole camra model to calculae rays from camere to each pixels
"""


def calculate_rays(
    height: int, width: int, focal_len: int, pose: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Find origin and direction of rays through every pixel and camera origin.

    :param height: image height
    :param width: image width
    :param focal_len:
    :param pose:
    :return: rays origin, rays direction
    """
    # Create image pixel grid and push to pose device
    i, j = torch.meshgrid(
        torch.arange(width, dtype=torch.float32).to(pose),
        torch.arange(height, dtype=torch.float32).to(pose),
        indexing="ij",
    )

    i, j = i.transpose(-1, -2), j.transpose(-1, -2)

    # Calculate direction rays for each pixel
    directions = torch.stack(
        [
            (i - width * 0.5) / focal_len,
            -(j - height * 0.5) / focal_len,
            -torch.ones_like(i),
        ],
        dim=-1,
    )

    # Apply camera pose to directions
    rays_direction = torch.sum(directions[..., None, :] * pose[:3, :3], dim=-1)

    # Origin is same for all directions (the optical center)
    rays_origin = pose[:3, -1].expand(rays_direction.shape)

    return rays_origin, rays_direction
