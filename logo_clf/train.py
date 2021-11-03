import argparse
from utils import read_yaml, update_config

import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger

from callback import *
from datamodule import LogoDataModule
from lightning_module import LogoLightningModule


def train(config, wandb=False):
    lightning_module = LogoLightningModule(config['lightning_module'])
    datamodule = LogoDataModule(config['datamodule'])
    trainer_params = config['trainer']

    callbacks = []
    ckpt_callback = checkpoint_callback(
        dir_path="ckpt", monitor="avg_val_loss", save_top_k=-1
    )
    callbacks.append(ckpt_callback)
    if wandb:
        wandb_callback = LogoImageCallback(datamodule, label_path=config['datamodule']['dataset']['label_path'], data_path=config['datamodule']['dataset']['data_path'])
        callbacks.append(wandb_callback)

        wandb_logger = WandbLogger(project="Logo_vienna_code_classification")
        trainer_params.update({'logger': wandb_logger})
    
    trainer = pl.Trainer(callbacks=callbacks, **trainer_params)
    trainer.fit(lightning_module, datamodule)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default=None, help="enter user config yaml path")
    parser.add_argument("--wandb", action="store_true", help="wandb")
    parser.add_argument("--device", type=str, default="0", choices=["0", "1", "cpu"], help="device")

    args = parser.parse_args()

    default_config = read_yaml("config/default.yaml")
    config = update_config(default_config, args.config)
    if args.device == 'cpu':
        config['trainer'].pop('gpus')
    else:
        config['trainer']['gpus'] = int(args.device)
    train(config, args.wandb)