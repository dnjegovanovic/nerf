from typing import Optional, Tuple

import torch
import torch.nn as nn


class NeRF(nn.Module):
    def __init__(
        self,
        d_input: int = 3,
        n_layers: int = 8,
        d_filter: int = 256,
        skip_con: Tuple[int] = (4,),
        d_viewdirs: Optional[int] = None,
    ):
        super().__init__()

        self.d_input = d_input
        self.n_layers = n_layers
        self.d_filter = d_filter
        self.skip_con = skip_con
        self.d_viewdirs = d_viewdirs

        # Create model

        self.layers = nn.ModuleList()
        self.layers.append(nn.Linear(self.d_input, self.d_filter))
        for i in range(self.n_layers):
            if i in self.skip_con:
                self.layers.append(
                    nn.Linear(self.d_filter + self.d_input, self.d_filter)
                )
            else:
                self.layers.append(nn.Linear(self.d_filter, self.d_filter))

        # Bottleneck layers
        if self.d_viewdirs is not None:
            # If using viewdirs, split alpha and RGB
            self.alpha_out = nn.Linear(self.d_filter, 1)
            self.rgb_filters = nn.Linear(self.d_filter, self.d_filter)
            self.branch = nn.Linear(self.d_filter + self.d_viewdirs, self.d_filter // 2)
            self.output = nn.Linear(self.d_filter // 2, 3)
        else:
            # If no viewdirs, use simpler output RGB and Alpha
            self.output = nn.Linear(self.d_filter, 4)

    def forward(
        self, x: torch.Tensor, viewdirs: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        # Cannot use viewdirs if instantiated with d_viewdirs = None
        if self.d_viewdirs is None and viewdirs is not None:
            raise ValueError("Cannot input x_direction if d_viewdirs was not given.")

            # Apply forward pass up to bottleneck
            x_input = x
            for i, layer in enumerate(self.layers):
                x = self.act(layer(x))
                if i in self.skip_con:
                    x = torch.cat([x, x_input], dim=-1)

            # Apply bottleneck
            if self.d_viewdirs is not None:
                # Split alpha from network output
                alpha = self.alpha_out(x)

                # Pass through bottleneck to get RGB
                x = self.rgb_filters(x)
                x = torch.concat([x, viewdirs], dim=-1)
                x = self.act(self.branch(x))
                x = self.output(x)

                # Concatenate alphas to output
                x = torch.concat([x, alpha], dim=-1)
            else:
                # Simple output
                x = self.output(x)
            return x
