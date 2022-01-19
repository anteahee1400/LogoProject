from pytorch_lightning.callbacks import ModelCheckpoint

def checkpoint_callback(dir_path=None, name=None, monitor="avg_val_loss", **kwargs):
    # if monitor=="valid_accuracy_epoch":
    #     filename="{epoch:03d}-{valid_accuracy_epoch:.4f}" 
    # elif monitor=="valid_preck_epoch":
    #     filename="{epoch:03d}-{valid_preck_epoch:.4f}"
    # elif monitor=="valid_reck_epoch":
    #     filename="{epoch:03d}-{valid_reck_epoch:.4f}"
    # elif monitor=="valid_f1k_epoch":
    #     filename="{epoch:03d}-{valid_f1k_epoch:.4f}"
    # else:
    #     filename= "{epoch:03d}-{avg_val_loss:.4f}"
    if name == None:
        name = ""
    filename = name + "{epoch:03d}--{" + monitor + ":.4f}"
    callback = ModelCheckpoint(
        dirpath=dir_path, filename=filename, monitor=monitor, **kwargs
    )
    return callback