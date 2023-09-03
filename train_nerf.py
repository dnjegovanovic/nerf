import torch
import pytorch_lightning as pl
import pytorch_lightning.loggers as loggers

from argparse import ArgumentParser
from nerf_model.config.core import config
from nerf_model.modules.NeRFModules import NeRFModule
from nerf_model.models.EarlyStoppingMech import EarlyStopping
from pathlib import Path


def main(hparams):
    save_dir = Path("output") / hparams.test_name
    save_dir.mkdir(exist_ok=True, parents=True)
    logger = loggers.TensorBoardLogger(save_dir, name="logs", version=1, log_graph=True)

    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        monitor="val_loss",
        filename="model-{epoch:02d}-{val_loss:.5f}",
        save_top_k=2,
        mode="min",
        save_last=True,
    )

    early_stop_callback = pl.callbacks.EarlyStopping(
        monitor="val_loss",
        min_delta=0.0001,
        patience=50,
        verbose=False,
        mode="min",
    )
    
    #early_stop_callback = EarlyStopping(50)

    callbacks = [checkpoint_callback, early_stop_callback]

    # hparams.stochastic_weight_avg = False

    trainer = pl.Trainer(
        accelerator="gpu",
        devices=[hparams.gpu],
        logger=logger,
        callbacks=callbacks,
        check_val_every_n_epoch=hparams.check_val_every_n_epochs,
        benchmark=hparams.benchmark,
        max_epochs=hparams.max_epochs
    )

    model = NeRFModule(config.model_training_config)  # proslediti config
    resume = hparams.resume
    del hparams.resume
    trainer.fit(model, ckpt_path=resume)


def add_base_arguments(parser):
    parser.add_argument("--test_name", type=str, help="Test name", default="test_01")
    parser.add_argument("--resume", type=str, help="resume from checkpoint")
    parser.add_argument(
        "--gpu", type=int, default=0, help="specify device id of gpu to train on"
    )
    parser.add_argument(
        "--check-val-every-n-epochs",
        type=int,
        default=1,
        help="check-val-every-n-epochs",
    )
    parser.add_argument("--benchmark", type=bool, default=True, help="benchmark")
    parser.add_argument(
        "--max-epochs", type=int, default=100, help="number of epochs"
    )

    return parser


if __name__ == "__main__":
    parser = ArgumentParser(add_help=True)
    parser = add_base_arguments(parser)
    main(parser.parse_args())
