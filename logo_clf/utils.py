import json
import yaml
import inspect


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
    if datamodule is not None:
        for k1 in datamodule.keys():
            for k2 in datamodule[k1].keys():
                default["datamodule"][k1][k2] = datamodule[k1][k2]

    if lightning_module is not None:
        for k1 in lightning_module.keys():
            for k2 in lightning_module[k1].keys():
                default["lightning_module"][k1][k2] = lightning_module[k1][k2]

    if trainer is not None:
        for k in trainer.keys():
            default["trainer"][k] = trainer[k]

    return default
