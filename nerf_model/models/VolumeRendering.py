import torch
from torch import nn
from typing import Tuple


class VolumeRendering:
    def __init__(
        self,
        raw_noise_std: float = 0.0,
        white_bkgd: bool = False,
    ) -> None:
        self.raw_noise_std = raw_noise_std
        self.white_bkgd = white_bkgd

    def _cumprod_exclusive(self, tensor: torch.Tensor) -> torch.Tensor:
        """
        Mimick functionality of tf.math.cumprod(..., exclusive=True), as it isn't available in PyTorch.

        Args:
        tensor (torch.Tensor): Tensor whose cumprod (cumulative product, see `torch.cumprod`) along dim=-1
            is to be computed.
        Returns:
        cumprod (torch.Tensor): cumprod of Tensor along dim=-1, mimiciking the functionality of
            tf.math.cumprod(..., exclusive=True) (see `tf.math.cumprod` for details).
        """

        # Compute regular cumprod first (this is equivalent to `tf.math.cumprod(..., exclusive=False)`).
        cumprod = torch.cumprod(tensor, -1)

        # "Roll" the elements along dimension 'dim' by 1 element.
        cumprod = torch.roll(cumprod, 1, -1)

        # Replace the first element by "1" as this is what tf.cumprod(..., exclusive=True) does.
        cumprod[..., 0] = 1.0

        return cumprod

    def raw_2_outputs(
        self,
        raw: torch.Tensor,
        z_vals: torch.Tensor,
        rays_direction: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Convert the raw NeRF output into RGB and other maps.
        """
        # Difference between consecutive elements of `z_vals`. [n_rays, n_samples]
        dists = z_vals[..., 1:] - z_vals[..., :-1]
        dists = torch.cat([dists, 1e10 * torch.ones_like(dists[..., :1])], dim=-1)

        # Multiply each distance by the norm of its corresponding direction ray
        # to convert to real world distance (accounts for non-unit directions).
        dists = dists * torch.norm(rays_direction[..., None, :], dim=-1)

        # Add noise to model's predictions for density. Can be used to
        # regularize network during training (prevents floater artifacts).
        noise = 0.0
        if self.raw_noise_std > 0.0:
            noise = torch.randn(raw[..., 3].shape) * self.raw_noise_std

        # Predict density of each sample along each ray. Higher values imply
        # higher likelihood of being absorbed at this point. [n_rays, n_samples]
        alpha = 1.0 - torch.exp(-nn.functional.relu(raw[..., 3] + noise) * dists)

        # Compute weight for RGB of each sample along each ray. [n_rays, n_samples]
        # The higher the alpha, the lower subsequent weights are driven.
        weights = alpha * self._cumprod_exclusive(1.0 - alpha + 1e-10)

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
        if self.white_bkgd:
            rgb_map = rgb_map + (1.0 - acc_map[..., None])

        return rgb_map, depth_map, acc_map, weights
