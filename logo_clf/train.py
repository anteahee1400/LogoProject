import argparse

import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger

from callback import LogoImageCallback
from datamodule import LogoDataModule
from lightning_module import LogoLightningModule
from utils import read_yaml, update_config


# if you import wandb, then it will cause some confusions later. 
# dont use keyword or lib name like this
# def train(config, use_wandb=False):
def train(config, wandb=False):
    trainer_params = config['trainer']
    
    datamodule = LogoDataModule(config['datamodule'])
    lightning_module = LogoLightningModule(config['lightning_module'])
    
    callbacks = []
    ckpt_callback = checkpoint_callback(
        dir_path="ckpt", monitor="avg_val_loss", save_top_k=-1
    )
    callbacks.append(ckpt_callback)
    
    if wandb:
        wandb_callback = LogoImageCallback(
            datamodule, 
            label_path=config['datamodule']['dataset']['label_path'], 
            data_path=config['datamodule']['dataset']['data_path']
        )
        callbacks.append(wandb_callback)
        
        wandb_logger_setting = trainer_params.pop("wandb_logger")
        wandb_logger = WandbLogger(**wandb_logger_setting)
    
    trainer = pl.Trainer(callbacks=callbacks, logger=wandb_logger, **trainer_params)
    trainer.fit(lightning_module, datamodule)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default=None, help="enter user config yaml path")
    parser.add_argument("--wandb", action="store_true", help="wandb")
    parser.add_argument("--device", type=str, default=None, choices=["0", "1", "cpu"], help="device")
    args = parser.parse_args()

    default_config = read_yaml("config/default.yaml")
    config = update_config(default_config, args.config)
    
    if args.device is None:
        pass
    elif args.device == 'cpu':
        config['trainer'].pop('gpus')
    else:
        config['trainer']['gpus'] = int(args.device)

    train(config, args.wandb)
