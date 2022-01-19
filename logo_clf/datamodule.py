import pytorch_lightning as pl
from torch.utils.data import DataLoader

from logo_clf.dataloader import load_dataset, LogoDataset
from logo_clf.utils import get_kwargs_keys_from_method


class LogoDataModule(pl.LightningDataModule):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self._setup_configure()

    def _setup_configure(self):
        dataset_keys = get_kwargs_keys_from_method(LogoDataset.__init__)
        self.dataset_kwargs = {
            k: v for k, v in self.config["dataset"].items() if k in dataset_keys
        }
        self.dataset_kwargs['dataset_cls'] = self.config['dataset']['dataset_cls']

        dataloader_keys = get_kwargs_keys_from_method(DataLoader.__init__)
        self.dataloader_kwargs = {
            k: v for k, v in self.config["dataloader"].items() if k in dataloader_keys
        }

    def setup(self, stage=None):
        if stage == "fit" or stage is None:
            self.dataset_train = self._dataset(split="train")
            self.dataset_val = self._dataset(split="val")
            # self.dims = tuple(self.dataset_train[0][0].shape)

        if stage == "test" or stage is None:
            self.dataset_test = self._dataset(split="test")
            # self.dims = tuple(self.dataset_test[0][0].shape)

    def train_dataloader(self, shuffle: bool = False, *args, **kwargs):
        kwargs = self.dataloader_kwargs.copy()
        return DataLoader(self.dataset_train, **kwargs)

    def val_dataloader(self, shuffle: bool = False, *args, **kwargs):
        kwargs = self.dataloader_kwargs.copy()
        kwargs.pop("shuffle")
        return DataLoader(self.dataset_val, shuffle=shuffle, **kwargs)

    def test_dataloader(self, shuffle: bool = False, *args, **kwargs):
        kwargs = self.dataloader_kwargs.copy()
        kwargs.pop("shuffle")
        return DataLoader(self.dataset_test, shuffle=shuffle, **kwargs)

    def _dataset(self, split):
        kwargs = self.dataset_kwargs.copy()
        return load_dataset(kwargs, split=split)
