import numpy as np
import torch

from typing import Optional, Tuple


class VolumeSampling:
    def __init__(
        self,
        n_samples: int,
        perturb: Optional[bool] = True,
        inverse_depth: bool = False,
    ):
        self.n_samples = n_samples
        self.perturb = perturb
        self.inverse_depth = inverse_depth

    def stratified_sampling(
        self,
        rays_origin: torch.Tensor,
        rays_direction: torch.Tensor,
        near: float,
        far: float,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Sample along ray from regularly-spaced bins.
        """
        # Grab samples for space integration along ray
        t_vals = torch.linspace(0.0, 1.0, self.n_samples, device=rays_origin.device)
        if not self.inverse_depth:
            # Sample linearly between `near` and `far`
            z_vals = near * (1.0 - t_vals) + far * (t_vals)
        else:
            # Sample linearly in inverse depth (disparity)
            z_vals = 1.0 / (1.0 / near * (1.0 - t_vals) + 1.0 / far * (t_vals))

        if self.perturb:
            mids = 0.5 * (z_vals[1:] + z_vals[:-1])
            upper = torch.concat([mids, z_vals[-1:]], dim=-1)
            lower = torch.concat([z_vals[:1], mids], dim=-1)
            t_rand = torch.rand([self.n_samples], device=z_vals.device)
            z_vals = lower + (upper - lower) * t_rand

        z_vals = z_vals.expand(list(rays_origin.shape[:-1]) + [self.n_samples])
        # Apply scale from `rays_direction` and offset from `rays_origin` to samples
        # pts: (width, height, n_samples, 3)

        pts = (
            rays_origin[..., None, :]
            + rays_direction[..., None, :] * z_vals[..., :, None]
        )

        return pts, z_vals
