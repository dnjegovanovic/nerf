import pytorch_lightning as pl

import torch
import torch.optim as optim
import torch.utils.data as data

from typing import List, Callable

from models import NeRFModel, PositionalEncoder


class NeRFModule(pl.LightningModule):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.__dict__.update(kwargs)
        self.n_layers = self.n_layers
        self.d_filter = self.d_filter
        self.skip = self.skip
        self.d_viewdirs = self.d_viewdirs

    def _setup_architecture(self):
        # set up positiona ecnoder
        self.encoder = PositionalEncoder.PositionalEncoder(
            self.d_input, self.n_freqs, self.log_spec
        )
        self.encode = lambda x: self.encoder(x)

        if self.use_viewdirs:
            self.encoder_viewdirs = PositionalEncoder.PositionalEncoder(
                self.d_input, self.n_freqs_views, self.log_spec
            )
            self.encode_viewdirs = lambda x: self.encoder_viewdirs(x)
            self.d_viewdirs = self.encode_viewdirs.d_output
        else:
            self.encode_viewdirs = None
            self.d_viewdirs = None

        # NeRF model
        self.model = NeRFModel(
            d_input=self.encoder.d_output,
            n_layers=self.n_layers,
            d_filter=self.d_filter,
            skip=self.skip,
            d_viewdirs=self.d_viewdirs,
        )

    def _get_chunks(
        self, inputs: torch.Tensor, chunk_size: int = 2**15
    ) -> List[torch.Tensor]:
        r"""
        Divide an input into chunks.
        """
        return [
            inputs[i : i + chunk_size] for i in range(0, inputs.shape[0], chunk_size)
        ]

    def _prepare_chunks(
        self,
        points: torch.Tensor,
        encoding_function: Callable[[torch.Tensor], torch.Tensor],
        chunk_size: int = 2**15,
    ) -> List[torch.Tensor]:
        r"""
        Encode and chunkify points to prepare for NeRF model.
        """
        points = points.reshape((-1, 3))
        points = encoding_function(points)
        points = self._get_chunks(points, chunksize=chunk_size)
        return points

    def _prepare_viewdirs_chunks(
        self,
        points: torch.Tensor,
        rays_d: torch.Tensor,
        encoding_function: Callable[[torch.Tensor], torch.Tensor],
        chunk_size: int = 2**15,
    ) -> List[torch.Tensor]:
        r"""
        Encode and chunkify viewdirs to prepare for NeRF model.
        """
        # Prepare the viewdirs
        viewdirs = rays_d / torch.norm(rays_d, dim=-1, keepdim=True)
        viewdirs = viewdirs[:, None, ...].expand(points.shape).reshape((-1, 3))
        viewdirs = encoding_function(viewdirs)
        viewdirs = self._get_chunks(viewdirs, chunksize=chunk_size)

        return viewdirs

    def forward(self, x):
        raise NotImplementedError
        # return

    def validation_step(self, sample, batch_idx):
        raise NotImplementedError

    def train_dataloader(self):
        return data.DataLoader(
            self.train_set,
            batch_size=self.batch_size,
            shuffle=True,
            drop_last=True,
            num_workers=8,
            pin_memory=True,
            persistent_workers=True,
            timeout=30,
        )

    def val_dataloader(self):
        return data.DataLoader(
            self.val_set,
            batch_size=self.batch_size,
            shuffle=False,
            drop_last=False,
            num_workers=8,
            pin_memory=True,
            persistent_workers=True,
            timeout=30,
        )

    def configure_optimizers(self):
        opt_gen = optim.Adam(
            [
                {
                    "params": self.regressor.parameters(),
                },
                {
                    "params": self.decoder.parameters(),
                },
            ],
            lr=self.lr,
        )

        opt_disc = optim.Adam(self.discriminator.parameters(), lr=self.lr)

        scheduler = optim.lr_scheduler.CosineAnnealingLR(opt_gen, self.max_epochs)
        return [opt_gen, opt_disc], {"scheduler": scheduler}
