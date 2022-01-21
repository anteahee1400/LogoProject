import argparse

import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger

from logo_clf.utils import read_yaml, update_config
from logo_clf.callback import *
from logo_clf.datamodule import LogoDataModule
from logo_clf.lightning_module import LogoLightningModule


def train(config, wandb=False):
    lightning_module = LogoLightningModule(config["lightning_module"])
    datamodule = LogoDataModule(config["datamodule"])
    trainer_params = config["trainer"]
    
    callbacks = []
    ckpt_callback = checkpoint_callback(config)
    callbacks.append(ckpt_callback)

    wandb_logger_setting = trainer_params.pop("wandb_logger")
    if wandb:
        wandb_logger = WandbLogger(**wandb_logger_setting)
        trainer_params.update({"logger": wandb_logger})

        wandb_callback_params = config["callback"]
        wandb_callback_cls = wandb_callback_params.pop("callback_cls")
        wandb_callback = eval(wandb_callback_cls)(datamodule, **wandb_callback_params)
        callbacks.append(wandb_callback)

    trainer = pl.Trainer(callbacks=callbacks, **trainer_params)
    trainer.fit(lightning_module, datamodule)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", type=str, default=None, help="enter user config yaml path"
    )
    parser.add_argument("--wandb", action="store_true", help="wandb")
    parser.add_argument(
        "--device", type=str, default=None, choices=["0", "1", "cpu"], help="device"
    )

    args = parser.parse_args()

    default_config = read_yaml("config/default.yaml")
    config = update_config(default_config, args.config)

    if args.device is None:
        pass
    elif args.device == "cpu":
        config["trainer"].pop("gpus")
    else:
        config["trainer"]["gpus"] = [int(args.device)]

    train(config, args.wandb)
