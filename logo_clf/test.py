import argparse
import pandas as pd
from utils import read_yaml, update_config

import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger

from callback import *
from datamodule import LogoDataModule
from lightning_module import LogoLightningModule


def test(config, wandb=False, ckpt=None):
    lightning_module = LogoLightningModule(config['lightning_module'])
    if ckpt is not None:
        lightning_module = lightning_module.load_from_checkpoint(checkpoint_path=ckpt)
    datamodule = LogoDataModule(config['datamodule'])
    trainer_params = config['trainer']

    callbacks = []
    if wandb:
        wandb_callback = LogoImageCallback(datamodule, label_path=config['datamodule']['dataset']['label_path'], data_path=config['datamodule']['dataset']['data_path'])
        callbacks.append(wandb_callback)

        wandb_logger = WandbLogger(project="Logo_vienna_code_classification")
        trainer_params.update({'logger': wandb_logger})
    
    trainer = pl.Trainer(callbacks=callbacks, **trainer_params)
    result = trainer.test(lightning_module, datamodule)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default=None, help="enter user config yaml path")
    parser.add_argument("--wandb", action="store_true", help="wandb")
    parser.add_argument("--device", type=str, default=None, choices=["0", "1", "cpu"], help="device")
    parser.add_argument("--ckpt", type=str, default=None, help="checkpoint")

    args = parser.parse_args()

    default_config = read_yaml("config/default.yaml")
    config = update_config(default_config, args.config)
    
    if args.device is None:
        pass
    elif args.device == 'cpu':
        config['trainer'].pop('gpus')
    else:
        config['trainer']['gpus'] = int(args.device)
    
    result = test(config, args.wandb, args.ckpt)
    pd.DataFrame.from_dict(result).to_csv("result.csv")
    