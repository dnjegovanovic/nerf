
import torch
import torch.nn as nn


class PositionalEncoder(nn.Module):
    """
    Positional encoding using sin-cosine to map input poinst to higher dimensional space
    """

    def __init__(self, dim_input: int, n_freqs: int, log_space: bool = False):
        """
        :param dim_input:
        :param n_freqs:
        :param log_space:
        """
        super().__init__()
        self.dim_input = dim_input  # Input dimension vec
        self.n_freqs = n_freqs  # Number of encoded frequencies
        self.log_space = log_space  # use log scale or linear

        self.dim_output = dim_input * (1 + 2 * self.n_freqs)  # Output dimenssion
        self.embed_fun = [lambda x: x]

        # Define frequencies in either linear or log scale
        # From paper
        if self.log_space:
            self.freq_bands = 2.0 ** torch.linspace(0.0, self.n_freqs - 1, self.n_freqs)
        else:
            self.freq_bands = torch.linspace(
                2.0**0.0, 2.0 ** (self.n_freqs - 1), self.n_freqs
            )

        # applay sin and cos
        self._applay_freq()

    def _applay_freq(self):
        for f in self.freq_bands:
            self.embed_fun.append(lambda x, freq=f: torch.sin(x * f))
            self.embed_fun.append(lambda x, freq=f: torch.cos(x * f))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.concat([fn(x) for fn in self.embed_fun], dim=-1)
