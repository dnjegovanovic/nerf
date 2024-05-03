from typing import Optional, Tuple

import torch
import torch.nn as nn

from nerf_app.utils.cumprod_exclusive import cumprod_exclusive


class VolumeRendering(nn.Module):
    """
    To convert raw NeRF outputs into an image, we utilize the volume integration method described in Equations 1-3 of Section 4 in the paper.
    """

    def __init__(self, config: dict = None):
        super().__init__()
        self.config = config

    @staticmethod
    def stratified_sampling(
        rays_origin: torch.Tensor,
        rays_direction: torch.Tensor,
        near_plane: float,
        far_plane: float,
        num_samples: int,
        perturb: Optional[bool] = True,
        inverse_depth: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Sample along ray from regularly-spaced bins.
        :param rays_origin:
        :param rays_direction:
        :param near_plane:
        :param far_plane:
        :param num_samples:
        :param perturb:
        :param inverse_depth:
        :return: Tuple[torch.Tensor, torch.Tensor]:
        """

        # Grab samples for space integration along ray
        # TODO: for "near" and "far" (instead of 0. and 1.)
        space_samples = torch.linspace(0.0, 1.0, num_samples, device=rays_origin.device)
        if not inverse_depth:
            # Sample linearly between `near` and `far`
            z_vals = near_plane * (1.0 - space_samples) + far_plane * space_samples
        else:
            # Sample linearly in inverse depth (disparity)
            z_vals = 1.0 / (
                1.0 / near_plane * (1.0 - space_samples)
                + 1.0 / far_plane * space_samples
            )

        # Draw uniform samples from bins along ray
        # Get intervals between samples.
        if perturb:
            mids = 0.5 * (z_vals[1:] + z_vals[:-1])  # mean value of samples on ray
            upper = torch.concat([mids, z_vals[-1:]], dim=-1)  # Upper bound
            lower = torch.concat([z_vals[:1], mids], dim=-1)  # lower bound
            val_rand = torch.rand([num_samples], device=z_vals.device)  # Random value
            z_vals = lower + (upper - lower) * val_rand  # final samples

        z_vals = z_vals.expand(list(rays_origin.shape[:-1]) + [num_samples])

        # Apply scale from `rays_d` and offset from `rays_o` to samples
        # pts -> (width, height, n_samples, 3)
        pts = (
            rays_origin[..., None, :]
            + rays_direction[..., None, :] * z_vals[..., :, None]
        )

        return pts, z_vals

    @staticmethod
    def volume_integration(
        raw: torch.Tensor,
        z_vals: torch.Tensor,
        rays_d: torch.Tensor,
        raw_noise_std: float = 0.0,
        white_bkgd: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Convert the raw NeRF output into RGB and other maps.
        """

        # Difference between consecutive elements of `z_vals`. [n_rays, n_samples]
        dists = z_vals[..., 1:] - z_vals[..., :-1]
        dists = torch.cat([dists, 1e10 * torch.ones_like(dists[..., :1])], dim=-1)

        # Multiply each distance by the norm of its corresponding direction ray
        # to convert to real world distance (accounts for non-unit directions).
        dists = dists * torch.norm(rays_d[..., None, :], dim=-1)

        # Add noise to model's predictions for density. Can be used to
        # regularize network during training (prevents floater artifacts).
        noise = 0.0
        if raw_noise_std > 0.0:
            noise = torch.randn(raw[..., 3].shape) * raw_noise_std

        # Predict density of each sample along each ray. Higher values imply
        # higher likelihood of being absorbed at this point. [n_rays, n_samples]
        alpha = 1.0 - torch.exp(-nn.functional.relu(raw[..., 3] + noise) * dists)

        # Compute weight for RGB of each sample along each ray. [n_rays, n_samples]
        # The higher the alpha, the lower subsequent weights are driven.
        weights = alpha * cumprod_exclusive(1.0 - alpha + 1e-10)

        # Compute weighted RGB map.
        rgb = torch.sigmoid(raw[..., :3])  # [n_rays, n_samples, 3]
        rgb_map = torch.sum(weights[..., None] * rgb, dim=-2)  # [n_rays, 3]

        # Estimated depth map is predicted distance.
        depth_map = torch.sum(weights * z_vals, dim=-1)

        # Disparity map is inverse depth.
        disp_map = 1.0 / torch.max(
            1e-10 * torch.ones_like(depth_map), depth_map / torch.sum(weights, -1)
        )

        # Sum of weights along each ray. In [0, 1] up to numerical error.
        acc_map = torch.sum(weights, dim=-1)

        # To composite onto a white background, use the accumulated alpha map.
        if white_bkgd:
            rgb_map = rgb_map + (1.0 - acc_map[..., None])

        return rgb_map, depth_map, acc_map, weights
