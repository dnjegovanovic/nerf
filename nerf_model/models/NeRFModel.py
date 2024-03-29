import torch
from torch import nn
from typing import Tuple, Optional


class NeRFModel(nn.Module):
    """_summary_

    Here we define the NeRF model, which consists primarily of a `ModuleList` of `Linear` layers, 
    separated by non-linear activation functions and the occasional residual connection.
    This implementation is based on Section 3 of the original "NeRF: Representing Scenes as
    Neural Radiance Fields for View Synthesis" paper and uses the same defaults.
    """

    def __init__(
        self,
        d_input: int = 3,
        n_layers: int = 8,
        d_filter: int = 256,
        skip: Tuple[int] = (4,),
        d_viewdirs: Optional[int] = None,
        *args,
        **kwargs
    ) -> None:
        super().__init__(*args, **kwargs)

        self.d_input = d_input
        self.n_layers = n_layers
        self.skip = skip
        self.act_fun = nn.functional.relu
        self.d_viewdirs = d_viewdirs
        self.d_filter = d_filter

        self._setup_architecture()

    def _setup_architecture(self):
        self.layers = nn.ModuleList(
            [nn.Linear(self.d_input, self.d_filter)]
            + [
                nn.Linear(self.d_filter + self.d_input, self.d_filter)
                if i in self.skip
                else nn.Linear(self.d_filter, self.d_filter)
                for i in range(self.n_layers - 1)
            ]
        )

        if self.d_viewdirs is not None:
            # If using viewdirs, split alpha and RGB
            self.alpha_chanel_out = nn.Linear(self.d_filter, 1)
            self.rgb_chanel = nn.Linear(self.d_filter, self.d_filter)
            self.branch = nn.Linear(self.d_filter + self.d_viewdirs, self.d_filter // 2)
            self.output = nn.Linear(self.d_filter // 2, 3)
        else:
            # If no viewdirs, use simpler output
            self.output = nn.Linear(self.d_filter, 4)

    def forward(self, x: torch.Tensor, viewdirs: Optional[torch.Tensor] = None):
        """
        Forward pass with optional view direction.
        """
        # Cannot use viewdirs if instantiated with d_viewdirs = None

        if self.d_viewdirs is None and viewdirs is not None:
            raise ValueError("Cannot input x_direction if d_viewdirs was not given.")

        # Apply forward pass up to bottleneck

        x_input = x
        for i, layer in enumerate(self.layers):
            x = self.act_fun(layer(x))
            if i in self.skip:
                x = torch.cat([x, x_input], dim=-1)

        # Apply bottleneck
        if self.d_viewdirs is not None:
            # Split alpha from network output
            alpha = self.alpha_chanel_out(x)

            # Pass through bottleneck to get RGB
            x = self.rgb_chanel(x)
            x = torch.concat([x, viewdirs], dim=-1)
            x = self.act_fun(self.branch(x))
            x = self.output(x)

            # Concatenate alphas to output
            x = torch.concat([x, alpha], dim=-1)
        else:
            # Simple output
            x = self.output(x)
        return x
