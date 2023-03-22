import torch
from torch import nn


class PositionalEncoder(nn.Module):
    """_summary_
    Positional encoding using sin-cosine to map input poinst to higher dimensional space
    """

    def __init__(self, d_input: int, num_freqs: int, log_space: bool = False) -> None:
        super().__init__()

        self.d_input = d_input
        self.num_freqs = num_freqs
        self.log_space = log_space
        self.d_output = d_input * (1 + 2 * self.num_freqs)
        self.embeded_fun = [lambda x: x]

        # define freq in either linear or log scale
        if self.log_space:
            self.freq_band = 2.0 ** torch.linspace(0.0, self.num_freqs - 1, self.num_freqs)
        else:
            self.freq_band = torch.linspace(
                2.0**0.0, 2.0 ** (self.num_freqs - 1), self.num_freqs
            )

        # applay sin and cos
        self._applay_freq()

    def _applay_freq(self):
        for freq in self.freq_band:
            self.embeded_fun.append(lambda x, freq=freq: torch.sin(x * freq))
            self.embeded_fun.append(lambda x, freq=freq: torch.cos(x * freq))

    def forward(self, x):
        return torch.concat([fn(x) for fn in self.embeded_fun], dim=-1)
