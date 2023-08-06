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

    def _sample_pdf(self, weights: torch.Tensor, bins: torch.Tensor) -> torch.Tensor:
        """
        Apply inverse transform sampling to a weighted set of points.
        """
        # Normalize weights to get PDF.
        pdf = (weights + 1e-5) / torch.sum(
            weights + 1e-5, -1, keepdims=True
        )  # [n_rays, weights.shape[-1]]

        # Convert PDF to CDF.
        cdf = torch.cumsum(pdf, dim=-1)  # [n_rays, weights.shape[-1]]
        cdf = torch.concat(
            [torch.zeros_like(cdf[..., :1]), cdf], dim=-1
        )  # [n_rays, weights.shape[-1] + 1]

        # Take sample positions to grab from CDF. Linear when perturb == 0.
        if not self.perturb:
            u = torch.linspace(0.0, 1.0, self.n_samples, device=cdf.device)
            u = u.expand(list(cdf.shape[:-1]) + [self.n_samples])  # [n_rays, n_samples]
        else:
            u = torch.rand(
                list(cdf.shape[:-1]) + [self.n_samples], device=cdf.device
            )  # [n_rays, n_samples]

        # Find indices along CDF where values in u would be placed.
        u = u.contiguous()  # Returns contiguous tensor with same values.
        inds = torch.searchsorted(cdf, u, right=True)  # [n_rays, n_samples]

        # Clamp indices that are out of bounds.
        below = torch.clamp(inds - 1, min=0)
        above = torch.clamp(inds, max=cdf.shape[-1] - 1)
        inds_g = torch.stack([below, above], dim=-1)  # [n_rays, n_samples, 2]

        # Sample from cdf and the corresponding bin centers.
        matched_shape = list(inds_g.shape[:-1]) + [cdf.shape[-1]]
        cdf_g = torch.gather(
            cdf.unsqueeze(-2).expand(matched_shape), dim=-1, index=inds_g
        )
        bins_g = torch.gather(
            bins.unsqueeze(-2).expand(matched_shape), dim=-1, index=inds_g
        )

        # Convert samples to ray length.
        denom = cdf_g[..., 1] - cdf_g[..., 0]
        denom = torch.where(denom < 1e-5, torch.ones_like(denom), denom)
        t = (u - cdf_g[..., 0]) / denom
        samples = bins_g[..., 0] + t * (bins_g[..., 1] - bins_g[..., 0])

        return samples  # [n_rays, n_samples]

    def sample_hierarchical(
        self,
        rays_origin: torch.Tensor,
        rays_direction: torch.Tensor,
        z_vals: torch.Tensor,
        weights: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Apply hierarchical sampling to the rays.
        """
        # Draw samples from PDF using z_vals as bins and weights as probabilities.
        z_vals_mid = 0.5 * (z_vals[..., 1:] + z_vals[..., :-1])
        new_z_samples = self._sample_pdf(
            z_vals_mid, weights[..., 1:-1], self.n_samples, perturb=self.perturb
        )
        new_z_samples = new_z_samples.detach()

        # Resample points from ray based on PDF.
        z_vals_combined, _ = torch.sort(
            torch.cat([z_vals, new_z_samples], dim=-1), dim=-1
        )
        pts = (
            rays_origin[..., None, :]
            + rays_direction[..., None, :] * z_vals_combined[..., :, None]
        )  # [N_rays, N_samples + n_samples, 3]
        return pts, z_vals_combined, new_z_samples
