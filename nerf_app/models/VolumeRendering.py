from typing import Optional, Tuple

import torch
import torch.nn as nn


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
