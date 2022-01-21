import os
import datetime
from pytorch_lightning.callbacks import ModelCheckpoint

def checkpoint_callback(config, **kwargs):
    name = config["lightning_module"]["model"]["name"] + "-"
    eval_metric = config["lightning_module"].get("evaluation")
    ckpt_monitor = "avg_val_loss"
    if eval_metric is not None:
        ckpt_monitor = eval_metric["items"][0]["kwargs"].get(
            "name", eval_metric["items"][0]["name"].split(".")[-1].lower()
        )
        ckpt_monitor = f"valid_{ckpt_monitor}_epoch"
    
    now = datetime.datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
    dir_path = os.path.join("ckpt", now)
    os.makedirs(dir_path, exist_ok=True)
    filename = name + "{epoch:03d}--{" + ckpt_monitor + ":.4f}"
    callback = ModelCheckpoint(
        dirpath=dir_path, filename=filename, monitor=ckpt_monitor, save_top_k=-1, **kwargs
    )
    return callback