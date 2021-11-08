import random
import numpy as np
from pathlib import Path
import cv2
from PIL import Image
import torch
from torchvision import transforms
from typing import List


class TensorTools:
    def to_tensor(self, value, **kwargs):
        return torch.tensor(value, **kwargs)


class ImageTools:
    def load_image(self, img_path, method="pil", rgb=True):
        if method == "pil":
            image = Image.open(img_path)
            if rgb:
                image = image.convert("RGB")
            else:
                image = image.convert("L")
        elif method == "cv2":
            image = cv2.imread(img_path)
            if rgb:
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            else:
                image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        return image

    def image_to_tensor(self, image):
        if isinstance(image, Image.Image):
            return self.pil_img_to_tensor(image)
        elif isinstance(image, np.ndarray):
            return self.np_img_to_tensor(image)
        else:
            raise AttributeError()

    def pil_img_to_tensor(self, image: Image.Image, **kwargs):
        return transforms.ToTensor()(image)

    def np_img_to_tensor(self, image: np.ndarray, **kwargs):
        return transforms.ToTensor()(Image.fromarray(image))


def get_files_from_paths(paths, extensions):
    files = []
    for p in paths if isinstance(paths, list) else [paths]:
        p = Path(p)
        if p.is_dir():
            for ext in extensions if isinstance(extensions, list) else [extensions]:
                files.extend(p.glob(f"**/*.{ext}"))
        else:
            if p.name.split(".")[-1] in extensions:
                files.append(p)
            else:
                raise Exception(f"{p} does not exist")
    return files
    
def split_df(df, seed=7777):
    random.seed(seed)
    splits = []
    for i, row in df.iterrows():
        if random.random() > 0.2:
            splits.append("train")
        else:
            splits.append("val")
    return splits

def sort_df(df, col='path'):
    return df.sort_values(by=col).reset_index(drop=True)


class TransformParser:
    """
    transforms.RandomSizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225])
    """

    @staticmethod
    def parse(queries):
        if isinstance(queries, list):
            results = []
            for q in queries:
                transform = eval(q)
                if isinstance(transform, tuple):
                    results.extend(list(transform))
                else:
                    results.append(transform)

            return transforms.Compose(results)

        else:
            return None


def parse_augmentation_texts(transform_texts: List[str]):
    """ """
    if transform_texts not in [None, [None]]:
        result = []
        for each in transform_texts:
            if isinstance(each, str):
                each = [each]
            result.append(TransformParser.parse(each))
        return result
    return [None]


def parse_transform_texts(transform_texts: List[str]):
    """ """
    return TransformParser.parse(transform_texts)
