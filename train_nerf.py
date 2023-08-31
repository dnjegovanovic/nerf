import torch
import pytorch_lightning as pl
import pytorch_lightning.loggers as loggers

from argparse import ArgumentParser

from nerf_model.config.core import config
from nerf_model.modules.NeRFModules import NeRFModule

from pathlib import Path


def main(hparams):
    save_dir = Path("output") / hparams.test_name
    save_dir.mkdir(exist_ok=True, parents=True)
    logger = loggers.TensorBoardLogger(save_dir, name="logs", version=1, log_graph=True)

    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        monitor="val_loss",
        filename="model-{epoch:02d}-{val_loss:.5f}",
        save_top_k=3,
        mode="min",
        save_last=True,
    )
    best_callback = pl.callbacks.ModelCheckpoint(
        monitor="val_loss",
        filename="best",
        save_top_k=1,
        mode="min",
        save_last=False,
    )
    early_stop_callback = pl.callbacks.EarlyStopping(
        monitor="val_loss",
        min_delta=0.001,
        patience=30,
        verbose=False,
        mode="min",
    )

    callbacks = [checkpoint_callback, best_callback, early_stop_callback]

    hparams.check_val_every_n_epochs = 1
    hparams.stochastic_weight_avg = False
    hparams.benchmark = True

    trainer = pl.Trainer.from_argparse_args(
        hparams,
        accelerator="gpu",
        devices=[hparams.gpu],
        logger=logger,
        callbacks=callbacks,
    )

    model = NeRFModule()  # proslediti config
    resume = hparams.resume
    del hparams.resume
    trainer.fit(model, ckpt_path=resume)
