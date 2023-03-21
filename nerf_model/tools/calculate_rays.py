import torch


def get_rays(
    height: int,
    width: int,
    focal_length: float,
    transform_matrix: torch.Tensor,
):
    """
    Get rays from spec positions.
    Find origin and direction of rays through every pixel and camera origin.
    """
    # Apply pinhole camera model to gather directions at each pixel

    i, j = torch.meshgrid(
        torch.arange(width, dtype=torch.float32).to(transform_matrix),
        torch.arange(height, dtype=torch.float32).to(transform_matrix),
        indexing="ij",
    )

    directions = torch.stack(
        [
            (i - width * 0.5) / focal_length,
            -(j - height * 0.5) / focal_length,
            -torch.ones_like(i),
        ],
        dim=-1,
    )

    # Apply camera pose to directions
    rays_direction = torch.sum(
        directions[..., None, :] * transform_matrix[:3, :3], dim=-1
    )

    # Origin is same for all directions (the optical center)
    rays_origin = transform_matrix[:3, -1].expand(rays_direction.shape)
    return rays_origin, rays_direction
