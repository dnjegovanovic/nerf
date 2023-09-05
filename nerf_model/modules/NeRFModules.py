import numpy as np

import pytorch_lightning as pl

import torch
import torch.optim as optim
import torch.utils.data as data

from typing import List, Callable

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d

from nerf_model.models import NeRFModel, PositionalEncoder
from nerf_model.models.VolumeSampling import VolumeSampling
from nerf_model.models.VolumeRendering import VolumeRendering
from nerf_model.dataset.lego_dataset import LegoDataset
from nerf_model.dataset.base_dataset import SplicedDataset, SplicedRays
from nerf_model.tools.calculate_rays import *

from nerf_model.tools.rendering import *
from nerf_model.tools.plot_crop_data import *

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class NeRFModule(pl.LightningModule):
    def __init__(self, config, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.__dict__.update(kwargs)
        self.config = config

        self.n_layers = config.model["n_layers"]
        self.d_filter = config.model["d_filter"]
        self.skip = config.model["skip"]
        self.use_fine_model = config.model["use_fine_model"]
        self.data_dir = config.data_dir

        self.train_psnrs = []
        self.val_psnrs = []

        self.iter_nums = 0
        self.iternums = []
        self.iternums_val = []
        self.iter_nums_val = 0

        # device = device

        self._setup_data()
        self._setup_architecture()
        self.test_data_loader = self.test_dataloader()

    def _setup_architecture(self):
        # set up positiona ecnoder
        self.encoder = PositionalEncoder.PositionalEncoder(
            self.config.encoder["d_input"],
            self.config.encoder["n_freqs"],
            self.config.encoder["log_space"],
        )
        self.encode = lambda x: self.encoder(x)

        if self.config.encoder["use_viewdirs"]:
            self.encoder_viewdirs = PositionalEncoder.PositionalEncoder(
                self.config.encoder["d_input"],
                self.config.encoder["n_freqs"],
                self.config.encoder["log_space"],
            )
            self.encode_viewdir = lambda x: self.encoder_viewdirs(x)
            self.d_viewdirs = self.encoder_viewdirs.d_output
        else:
            self.encode_viewdir = None
            self.d_viewdirs = None

        # NeRF model
        self.model = NeRFModel.NeRFModel(
            d_input=self.encoder.d_output,
            n_layers=self.n_layers,
            d_filter=self.d_filter,
            skip=self.skip,
            d_viewdirs=self.d_viewdirs,
        )

        self.model.to(device)
        self.model_params = list(self.model.parameters())

        if self.use_fine_model:
            self.fine_model = NeRFModel.NeRFModel(
                d_input=self.encoder.d_output,
                n_layers=self.n_layers,
                d_filter=self.d_filter,
                skip=self.skip,
                d_viewdirs=self.d_viewdirs,
            )
            self.fine_model.to(device)
            self.model_params = self.model_params + list(self.fine_model.parameters())
        else:
            self.fine_model = None

        self.volume_sampling = VolumeSampling(
            self.config.strf_samp_option["n_samples"],
            self.config.hierarchical_sampling["n_samples_hierarchical"],
            self.config.strf_samp_option["perturb"],
            self.config.strf_samp_option["inverse_depth"],
        )

        self.volume_rendering = VolumeRendering(
            self.config.hierarchical_sampling["raw_noise_std"],
            self.config.hierarchical_sampling["white_bkgd"],
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
        points = self._get_chunks(points, chunk_size=chunk_size)
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
        viewdirs = self._get_chunks(viewdirs, chunk_size=chunk_size)

        return viewdirs

    def _setup_data(self):
        self.dataset = LegoDataset(self.data_dir)
        all_arays = np.array(self.dataset.get_all_rays())
        images = self.dataset.images
        # poses = self.dataset.poses
        num_of_data = images.shape[0]
        num_of_train_data = int(0.9 * num_of_data)

        self.train_data = SplicedRays(
            all_arays[:, 0][1:num_of_train_data],
            all_arays[:, 1][1:num_of_train_data],
            images[1:num_of_train_data],
        )
        self.val_data = SplicedRays(
            all_arays[:, 0][num_of_train_data:],
            all_arays[:, 1][num_of_train_data:],
            images[num_of_train_data:],
        )

        self.test_data = SplicedRays(
            all_arays[:, 0][:1],
            all_arays[:, 1][:1],
            images[0],
        )

    def _plot_data(
        self,
        rgb_predicted,
        testimg,
        train_psnrs,
        val_psnrs,
        outputs,
        n_samples,
        n_samples_hierarchical,
        iternums,
        i,
    ):
        if self.config.training_config["batch_size"] > 1:
            pred_img = (
                rgb_predicted.reshape(
                    [-1, self.dataset.img_height, self.dataset.img_width, 3]
                )[0]
                .detach()
                .cpu()
                .numpy()
            )
            test_img_one = (
                testimg.reshape(
                    [-1, self.dataset.img_height, self.dataset.img_width, 3]
                )[0]
                .detach()
                .cpu()
                .numpy()
            )
        else:
            pred_img = (
                rgb_predicted.reshape(
                    [self.dataset.img_height, self.dataset.img_width, 3]
                )
                .detach()
                .cpu()
                .numpy()
            )
            test_img_one = (
                testimg.reshape([self.dataset.img_height, self.dataset.img_width, 3])
                .detach()
                .cpu()
                .numpy()
            )

        # Plot example outputs
        fig, ax = plt.subplots(
            1, 4, figsize=(24, 4), gridspec_kw={"width_ratios": [1, 1, 1, 3]}
        )
        ax[0].imshow(pred_img)
        ax[0].set_title(f"Iteration: {i}")
        ax[1].imshow(test_img_one)
        ax[1].set_title(f"Target")
        ax[2].plot(range(0, i), train_psnrs, "r")
        # ax[2].plot(iternums, val_psnrs, "b")
        ax[2].set_title("PSNR (train=red, val=blue")
        z_vals_strat = outputs["z_vals_stratified"].view((-1, n_samples))
        z_sample_strat = z_vals_strat[z_vals_strat.shape[0] // 2].detach().cpu().numpy()
        if "z_vals_hierarchical" in outputs:
            z_vals_hierarch = outputs["z_vals_hierarchical"].view(
                (-1, n_samples_hierarchical)
            )
            z_sample_hierarch = (
                z_vals_hierarch[z_vals_hierarch.shape[0] // 2].detach().cpu().numpy()
            )
        else:
            z_sample_hierarch = None
        _ = plot_samples(z_sample_strat, z_sample_hierarch, ax=ax[3])
        ax[3].margins(0)
        plt.savefig(self.config.save_file + "/{}.png".format(i))
        plt.close(fig)

    def forward(self, x):
        if self.config.training_config["batch_size"] > 1:
            rays_o = x["rays_o"]
            rays_o = rays_o.reshape([-1, 3])
            rays_d = x["rays_d"]
            rays_d = rays_d.reshape([-1, 3])

        else:
            rays_o = x["rays_o"][0]
            rays_d = x["rays_d"][0]

            rays_o = rays_o.reshape([-1, 3])
            rays_d = rays_d.reshape([-1, 3])

        # Sample query points along each ray.
        query_points, z_vals = self.volume_sampling.stratified_sampling(
            rays_o,
            rays_d,
            self.config.strf_samp_option["near"],
            self.config.strf_samp_option["far"],
        )
        # Prepare batches.
        batches = self._prepare_chunks(
            query_points,
            self.encode,
            chunk_size=self.config.training_config["chunksize"],
        )
        if self.encode_viewdir is not None:
            batches_viewdirs = self._prepare_viewdirs_chunks(
                query_points,
                rays_d,
                self.encode_viewdir,
                chunk_size=self.config.training_config["chunksize"],
            )
        else:
            batches_viewdirs = [None] * len(batches)

        # Coarse model pass.
        # Split the encoded points into "chunks", run the model on all chunks, and
        # concatenate the results (to avoid out-of-memory issues).
        predictions = []
        for batch, batch_viewdirs in zip(batches, batches_viewdirs):
            predictions.append(self.model(batch, viewdirs=batch_viewdirs))
        raw = torch.cat(predictions, dim=0)
        raw = raw.reshape(list(query_points.shape[:2]) + [raw.shape[-1]])

        # Perform differentiable volume rendering to re-synthesize the RGB image.
        rgb_map, depth_map, acc_map, weights = self.volume_rendering.raw_2_outputs(
            raw, z_vals, rays_d
        )
        # rgb_map, depth_map, acc_map, weights = render_volume_density(raw, rays_o, z_vals)
        outputs = {"z_vals_stratified": z_vals}

        # Fine model pass.
        if self.config.hierarchical_sampling["n_samples_hierarchical"] > 0:
            # Save previous outputs to return.
            rgb_map_0, depth_map_0, acc_map_0 = rgb_map, depth_map, acc_map

            # Apply hierarchical sampling for fine query points.
            (
                query_points,
                z_vals_combined,
                z_hierarch,
            ) = self.volume_sampling.sample_hierarchical(
                rays_o, rays_d, z_vals, weights
            )

            # Prepare inputs as before.
            batches = self._prepare_chunks(
                query_points,
                self.encode,
                chunk_size=self.config.training_config["chunksize"],
            )
            if self.encode_viewdir is not None:
                batches_viewdirs = self._prepare_viewdirs_chunks(
                    query_points,
                    rays_d,
                    self.encode_viewdir,
                    chunk_size=self.config.training_config["chunksize"],
                )
            else:
                batches_viewdirs = [None] * len(batches)

            # Forward pass new samples through fine model.
            fine_model = self.fine_model if self.use_fine_model else self.model
            predictions = []
            for batch, batch_viewdirs in zip(batches, batches_viewdirs):
                predictions.append(fine_model(batch, viewdirs=batch_viewdirs))
            raw = torch.cat(predictions, dim=0)
            raw = raw.reshape(list(query_points.shape[:2]) + [raw.shape[-1]])

            # Perform differentiable volume rendering to re-synthesize the RGB image.
            rgb_map, depth_map, acc_map, weights = self.volume_rendering.raw_2_outputs(
                raw, z_vals_combined, rays_d
            )

            outputs["z_vals_hierarchical"] = z_hierarch
            outputs["rgb_map_0"] = rgb_map_0
            outputs["depth_map_0"] = depth_map_0
            outputs["acc_map_0"] = acc_map_0

            # Store outputs.
            outputs["rgb_map"] = rgb_map
            outputs["depth_map"] = depth_map
            outputs["acc_map"] = acc_map
            outputs["weights"] = weights

            self._training_sanity_check(outputs)

            return outputs

    def training_step(self, batch, batch_idx):
        outputs = self(batch)
        rgb_predicted = outputs["rgb_map"]
        tgt_images = batch["images"].reshape(-1, 3)
        loss = torch.nn.functional.mse_loss(rgb_predicted, tgt_images)

        self.log("train_loss", loss.item(), prog_bar=True)
        psnr = -10.0 * torch.log10(loss)
        self.log("train_psnr", psnr.item(), prog_bar=True)
        self.train_psnrs.append(psnr.item())

        self.iter_nums += 1
        self.iternums.append(self.iter_nums)

        return loss

    def validation_step(self, batch, batch_idx):
        outputs = self(batch)
        rgb_predicted = outputs["rgb_map"]
        tgt_images = batch["images"].reshape(-1, 3)
        loss = torch.nn.functional.mse_loss(rgb_predicted, tgt_images)

        self.log("val_loss", loss.item(), prog_bar=True)
        val_psnr = -10.0 * torch.log10(loss)
        self.log("val_psnr", val_psnr.item(), prog_bar=True)
        self.val_psnrs.append(val_psnr.item())
        self.iter_nums_val += 1
        self.iternums_val.append(self.iter_nums_val)

        if self.iter_nums % 40 == 0 and self.iter_nums != 0:
            # for sample in self.test_data_loader:
            #     outputs_test = self(sample)
            #     rgb_predicted_test = outputs_test["rgb_map"]
            #     tgt_images_test = sample["images"].reshape(-1, 3)
            self._plot_data(
                rgb_predicted,
                tgt_images,
                self.train_psnrs,
                self.val_psnrs,
                outputs,
                self.config.strf_samp_option["n_samples"],
                self.config.hierarchical_sampling["n_samples_hierarchical"],
                self.iternums,
                self.iter_nums,
            )

        return loss

    def train_dataloader(self):
        return data.DataLoader(
            self.train_data,
            batch_size=self.config.training_config["batch_size"],
            shuffle=True,
            drop_last=True,
            num_workers=8,
            pin_memory=True,
            persistent_workers=True,
            timeout=30,
        )

    def val_dataloader(self):
        return data.DataLoader(
            self.val_data,
            batch_size=self.config.training_config["batch_size"],
            shuffle=False,
            drop_last=False,
            num_workers=8,
            pin_memory=True,
            persistent_workers=True,
            timeout=30,
        )

    def test_dataloader(self):
        return data.DataLoader(
            self.test_data,
            batch_size=1,
            shuffle=False,
            drop_last=False,
            num_workers=2,
            pin_memory=True,
            persistent_workers=True,
            timeout=30,
        )

    def _training_sanity_check(self, outputs):
        # Check for any numerical issues.
        for k, v in outputs.items():
            if torch.isnan(v).any():
                print(f"! [Numerical Alert] {k} contains NaN.")
            if torch.isinf(v).any():
                print(f"! [Numerical Alert] {k} contains Inf.")

    def configure_optimizers(self):
        opt_model = optim.Adam(
            [
                {
                    "params": self.model_params,
                }
            ],
            lr=self.config.optimizer["lr"],
        )

        # scheduler = EarlyStopping(patience=50)
        # scheduler = optim.lr_scheduler.CosineAnnealingLR(opt_model, 100)
        return opt_model  # , {"scheduler": scheduler}
