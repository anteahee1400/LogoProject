import sys
import inspect
import pandas as pd
from pathlib import Path
from typing import Callable, List, Optional

import torch
from torch.utils.data import Dataset
from torchvision import transforms

from logo_clf.dataloader.utils import *


class LogoDataset(Dataset, TensorTools, ImageTools):
    
    def __init__(
        self,
        data_path: str,
        label_path: str = None,
        augmentations: List[Callable] = [None],
        transforms: Optional[Callable] = None,
        num_classes: int = None,
        split: str = "train",
        random_split: bool = False,
        label_col: str = "label_s",
        **kwargs,
    ):
        super().__init__()
        data_path = Path(data_path)
        self.data_path = data_path
        if data_path.name in ["train_folder", "test_folder"]:
            self.data_path = data_path.parent
        # self.train_path = self.data_path / "train_folder"
        # self.test_path = self.data_path / "test_folder"
        # self.train_data = get_files_from_paths(self.train_path, "jpg")
        # self.test_data = get_files_from_paths(self.test_path, "jpg")
        # if split == "test":
        #     self.data_files = self.test_data
        # else:
        #     self.data_files = self.train_data
        self.data_files = get_files_from_paths(self.data_path, "jpg")
        self.label_path = label_path
        self.augmentations = parse_augmentation_texts(augmentations)
        self.transforms = parse_transform_texts(transforms)
        self.num_classes = num_classes
        self.split = split
        self.random_split = random_split

        self.label_df = (
            pd.read_csv(label_path, low_memory=False)
            if label_path is not None
            else None
        )
        if self.label_df is not None:
            self.do_split()
            self.do_sort()
            self.label_df["label"] = self.label_df[label_col]
            self.load_mapping_dict()
            self.grouped_label_df = self.label_df.groupby(["path"]).count()
            self.len_grouped_label_df = len(self.grouped_label_df)

        if self.split != "train":
            self.augmentations = [None]
        if self.augmentations not in [None, [None]]:
            self.len_augmentations = len(self.augmentations)
        else:
            self.len_augmentations = 0

        self.return_label = True if isinstance(self.label_df, pd.DataFrame) else False

        assert self.label_df.path.unique().tolist() == [
            p.name for p in self.data_files
        ], "Labels and Data files don't match in order."

    def __len__(self):
        # return len(self.data_files)  # label_df
        # return self.len_grouped_label_df * (self.len_augmentations + 1)
        return 128

    def __getitem__(self, idx):

        augment_idx, idx = self.get_augmentation_idx(idx)
        image = self.load_image(self.data_files[idx])

        if augment_idx >= 0:
            image = self.augmentations[augment_idx](image)

        if self.transforms is not None:
            image = self.transforms(image)

        if self.return_label:
            if self.split == "test":
                labels = self.get_labels(idx)
                label = self.to_multilabel(labels)
            else:
                label = self.to_tensor(self.label_df.loc[idx]["label"])
        else:
            label = self.to_tensor(-1)

        return image, label

    def do_split(self):
        if "split" not in self.label_df.columns or self.random_split:
            splits = split_df(self.label_df)
            self.label_df["split"] = splits

        self.label_df = self.label_df[self.label_df.split == self.split].reset_index(
            drop=True
        )
        self.data_files = self.filter_out_by_split(self.data_files)

    def do_sort(self):
        self.label_df = sort_df(self.label_df)
        self.data_files = sorted(self.data_files, key=lambda x: x.name)

    def filter_out_by_split(self, data_files):
        data_df = pd.DataFrame({"abspath": data_files})
        data_df["path"] = data_df["abspath"].apply(lambda x: x.name)
        trg_imgs = list(set(data_df.path.unique()) & set(self.label_df.path.unique()))
        df = data_df[data_df.path.isin(trg_imgs)].reset_index(drop=True)
        return list(df.abspath.unique())

    def get_augmentation_idx(self, idx):
        augment_idx = -1
        if self.augmentations not in [None, [None]]:
            augment_idx = (idx // self.len_grouped_label_df) - 1
            idx = idx % self.len_grouped_label_df
        return augment_idx, idx

    def load_mapping_dict(self):
        if isinstance(self.num_classes, int):
            self.mapping_dict = {i: i for i in range(self.num_classes)}
        else:
            self.mapping_dict = {
                l: i
                for i, l in enumerate(sorted(list(self.label_df["label"].unique())))
            }

    def get_labels(self, idx):
        labels = list(
            self.label_df[self.label_df["path"] == self.grouped_label_df.index[idx]][
                "label"
            ]
        )
        labels = [self.mapping_dict[_label] for _label in labels]
        return labels

    def to_multilabel(self, labels):
        if isinstance(self.num_classes, int):
            label_tensor = [0] * self.num_classes
        else:
            label_tensor = [0] * len(self.label_df["label"].unique())

        for _label in labels:
            label_tensor[_label] = 1

        return self.to_tensor(label_tensor, dtype=torch.float32)


class LogoMultilabelDataset(LogoDataset):
    
    def __getitem__(self, idx):
    
        augment_idx, idx = self.get_augmentation_idx(idx)
        image = self.load_image(self.data_files[idx])

        if augment_idx >= 0:
            image = self.augmentations[augment_idx](image)

        if self.transforms is not None:
            image = self.transforms(image)

        if self.return_label:
            labels = self.get_labels(idx)
            label = self.to_multilabel(labels)
        else:
            label = self.to_tensor(-1)

        return (image, label)
    

def load_dataset_cls(cls_name):
    dataset_cls = None
    try:
        AVAILABLE_DATASETS = dict(
            inspect.getmembers(sys.modules[__name__], inspect.isclass)
        )
        dataset_cls = AVAILABLE_DATASETS[cls_name]
    except:
        print("Wrong dataset style")
    return dataset_cls


def load_dataset(dataset_dict, split):
    cls_name = dataset_dict.pop("dataset_cls")
    dataset_dict.update({"split": split})

    dataset_cls = load_dataset_cls(cls_name)

    return dataset_cls(**dataset_dict)


if __name__ == "__main__":
    transform = transforms.Compose(
        [
            transforms.Resize((416, 416)),
            transforms.RandomGrayscale(p=0.2),
            transforms.RandomInvert(p=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    dataset = LogoDataset(
        data_path="/home/ubuntu/datasets/LOGO/Logo_clf",
        label_path="/home/ubuntu/datasets/LOGO/Logo_clf/meta.csv",
        transforms=transform,
        split="train",
    )
