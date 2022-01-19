import json
import yaml
import inspect
import pandas as pd
import tqdm
import random
import torch
import torch.nn.functional as F



def read_yaml(path):
    """ """
    result = {}
    try:
        with open(path, encoding="cp949") as f:
            for param in yaml.load_all(f, Loader=yaml.FullLoader):
                result.update(param)

    except UnicodeDecodeError:
        with open(path, encoding="utf-8") as f:
            for param in yaml.load_all(f, Loader=yaml.FullLoader):
                result.update(param)

    return result


def to_yaml(path, info):
    with open(path, "w") as f:
        yaml.dump(info, f)


def read_json(path):
    try:
        with open(path, "r", encoding="cp949") as f:
            result = json.load(f)
    except UnicodeDecodeError:
        with open(path, "r", encoding="utf-8") as f:
            result = json.load(f)
    return result


def to_json(path, info):
    with open(path, "w") as f:
        json.dump(info, f, sort_keys=False, indent="\t", separators=(",", ": "))

def get_kwargs_keys_from_method(method):
    keys = []
    for k in inspect.signature(method).parameters.values():
        key = str(k).split(":")[0]
        key = key.split("=")[0]
        keys.append(key)
    return keys


def update_config(default, path):
    if path is None:
        return default
    update_dict = read_yaml(path)
    datamodule = update_dict.get("datamodule")
    lightning_module = update_dict.get("lightning_module")
    trainer = update_dict.get("trainer")
    callback = update_dict.get("callback")
    if datamodule is not None:
        for k1 in datamodule.keys():
            for k2 in datamodule[k1].keys():
                default["datamodule"][k1][k2] = datamodule[k1][k2]

    if lightning_module is not None:
        for k1 in lightning_module.keys():
            if not isinstance(lightning_module[k1], dict):
                default['lightning_module'][k1] = lightning_module[k1]
                continue
            for k2 in lightning_module[k1].keys():
                default["lightning_module"][k1][k2] = lightning_module[k1][k2]

    if trainer is not None:
        for k in trainer.keys():
            default["trainer"][k] = trainer[k]

    if callback is not None:
        for k in callback.keys():
            default["callback"][k] = callback[k]

    return default


def make_cv_meta_files(train_df, val_df):
    dfs = []
    for i in range(4):
        df = set_split_by_index(train_df, i)
        val_df["split"] = "train"
        df = pd.concat([df, val_df]).reset_index(drop=True)
        dfs.append(df)
    return dfs


def set_split_by_index(train_df, idx):
    split = []
    for i in tqdm.tqdm(range(train_df.shape[0])):
        if i % 4 == idx:
            split.append("val")
        else:
            split.append("train")
    train_df["split"] = split
    dfs = []
    for s in ["train", "val"]:
        dfs.append(train_df[train_df.split == s].reset_index(drop=True))
    return pd.concat(dfs).reset_index(drop=True)


def stratified_sampling(df):
    df_unique = df.groupby("path").first().reset_index()
    split_mapper = {}
    for l in tqdm.tqdm(df_unique.label.unique()):
        sub = df[df.label == l].sort_values(by="path").reset_index(drop=True)
        for p in sub.path.unique():
            prob = random.random()
            if prob > 0.3:
                split = "train"
            elif prob < 0.1:
                split = "test"
            else:
                split = "val"
            split_mapper[p] = split

    df["split"] = df["path"].apply(lambda x: split_mapper[x])
    return df


def split_multi_hot(label):
    try:
        batch_ids, batch_labels = torch.where(label == 1)
        labels = [[] for _ in range(label.shape[0])]
        for i, idx in enumerate(batch_ids):
            labels[idx].append(batch_labels[i])
        labels = [torch.stack(l) for l in labels]
        max_len = max([len(l) for l in labels])
        return labels, max_len
    except:
        return [torch.tensor([0])]*len(label), 1

def concat_multi_hot(label_unstack, num_classes):
    device = label_unstack[0].device
    unstack = [F.one_hot(label, num_classes).sum(dim=0) > 0 for label in label_unstack]
    return torch.stack(unstack).to(device)