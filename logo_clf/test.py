import argparse

import pandas as pd
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger

from callback import *
from datamodule import LogoDataModule
from lightning_module import LogoLightningModule
from utils import read_yaml, update_config


def test(config, use_wandb=False, ckpt=None):
    callbacks = []
    logger = None
    trainer_params = config["trainer"]
    logger_settings = trainer_params.pop("wandb_logger")
    
    # load model with lighitning Model
    lightning_module = LogoLightningModule(config["lightning_module"])
    if ckpt is not None:
        lightning_module.load_from_checkpoint(checkpoint_path=ckpt)
        
    # load datamodule
    datamodule = LogoDataModule(config["datamodule"])
    
    if use_wandb:
        wandb_callback = LogoImageCallback(
            datamodule,
            **config['callback']
        )
        callbacks.append(wandb_callback)
        logger = WandbLogger(**logger_settings)

    trainer = pl.Trainer(callbacks=callbacks, logger=logger, **trainer_params)
    result = trainer.test(lightning_module, datamodule)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", type=str, default=None, help="enter user config yaml path"
    )
    parser.add_argument("--wandb", action="store_true", help="wandb")
    parser.add_argument(
        "--device", type=str, default=None, choices=["0", "1", "cpu"], help="device"
    )
    parser.add_argument("--ckpt", type=str, default=None, help="checkpoint")

    args = parser.parse_args()

    default_config = read_yaml("config/default.yaml")
    config = update_config(default_config, args.config)

    if args.device is None:
        pass
    elif args.device == "cpu":
        config["trainer"].pop("gpus")
    else:
        config["trainer"]["gpus"] = [int(args.device)]

    result = test(config, args.wandb, args.ckpt)
    result_file = args.config.split('/')[-1].replace('.yaml', "")
    pd.DataFrame.from_dict(result).to_csv(f"result/{result_file}.csv")
