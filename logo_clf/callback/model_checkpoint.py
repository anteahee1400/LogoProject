from pytorch_lightning.callbacks import ModelCheckpoint

def checkpoint_callback(dir_path=None, **kwargs):
    callback = ModelCheckpoint(
        dirpath=dir_path, filename="{epoch:03d}-{avg_val_loss:.4f}", **kwargs
    )

    return callback