import matplotlib.pyplot as plt
import numpy as np
import tqdm

from PIL import Image


def code_to_str(df):
    """
    vienna code data type transformation (int -> str)
    Args:
        df: data frame to use
    Returns:
        df with transformated vienna code
    """
    if df["code_s"].dtype.name != "object":
        df["code_s"] = df["code_s"].apply(lambda x: str(x).zfill(6))
    if df["code_m"].dtype.name != "object":
        df["code_m"] = df["code_m"].apply(lambda x: str(x).zfill(4))
    if df["code_l"].dtype.name != "object":
        df["code_l"] = df["code_l"].apply(lambda x: str(x).zfill(2))
    return df


def mapper(df, key="code_s", value="desc_s"):
    """
    Args:
        df: data frame to use
        key: mapper's key to be used for searching (default: 'code_s')
        value: mapper's value to search for (default: 'desc_s')
    Returns:
        mapper
    """
    df = code_to_str(df)
    df = df.groupby(key).first().reset_index()
    return {row[key]: row[value] for i, row in df.iterrows()}


def bar(df, title, xlabel="index", ylabel="count"):
    """
    Args:
        df: data frame to use
        title: title of bar graph
        xlabel: label of x axis (default: 'index')
        ylabel: label of y axis (default: 'count')
    """
    plt.figure(figsize=(10, 6))
    plt.title(title)
    if xlabel == "index":
        plt.bar(df.index, df[ylabel])
    else:
        plt.bar(df[xlabel], df[ylabel])
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.show()


def class_balance(df, class_name="code_s", visual=False, **kwargs):
    """
    check the class distribution
    Args:
        df: data frame to use
        class_name: name of class to be specified (default: 'code_s')
        visual: if set True, the bar graph is displayed (default: False)
    Returns:
        grouped data frame (infomation of class name and it's count)
    """
    if class_name.endswith("s"):
        value = "desc_s_ko"
    elif class_name.endswith("m"):
        value = "desc_m_ko"
    else:
        value = "desc_l_ko"
    mapping_dict = mapper(df, key=class_name, value=value)
    df = code_to_str(df)
    df = (
        df.groupby(class_name)
        .count()
        .reset_index()
        .sort_values("path", ascending=False)
        .reset_index(drop=True)
    )
    df = df[[class_name, "path"]].rename(columns={"path": "count"})
    df["ratio"] = df["count"].apply(lambda x: round((x / df["count"].sum()) * 100, 3))
    df["desc"] = df[class_name].apply(lambda x: mapping_dict[x])
    if visual:
        bar(df, "Class Distribution", **kwargs)
    return df


def number_of_classes_by_image(df, visual=False):
    """
    check the distribution of the number of classes by image
    Args:
        df: data frame to use
        visual: if set True, the bar graph is displayed (default: False)
    Returns:
        grouped data frame (information of the number of classes, count of images, and the ratio of the number of classes)
    """
    df = (
        df.groupby("path")
        .count()
        .reset_index()
        .sort_values("label", ascending=False)
        .reset_index(drop=True)
    )
    df = df[["path", "label"]].rename(columns={"path": "count", "label": "num_classes"})
    df = (
        df.groupby("num_classes")
        .count()
        .reset_index()
        .sort_values("count", ascending=False)
        .reset_index(drop=True)
    )
    df = df[["num_classes", "count"]]
    df["percentage"] = df["count"] * 100 / df["count"].sum()
    df["percentage"] = df["percentage"].apply(lambda x: f"{x:.02f}%")
    if visual:
        bar(df.head(10), "Number of Classes by Image", xlabel="num_classes")
    return df


def average_width_and_height(df):
    """
    check the averate of image width and height
    Args:
        df: data frame to use
    Returns:
        average of image width, average of image height
    """
    return np.mean(df.width), np.mean(df.height)


def number_of_images(df):
    """
    check the number of images
    Args:
        df: data frame to use
    Returns:
        the number of images
    """
    return len(df.path.unique())


def median_image_ratio(df):
    """
    check the median of image ratio
    Args:
        df: data frame to use
    Returns:
        median of image width, median of image height
    """
    df = df[["width", "height"]].describe()
    return df.loc["50%"].width, df.loc["50%"].height


def annotation_heatmap(df, split="train", label="all"):
    """
    draw annotation (bounding boxes of the target objects of images) heatmap
    Args:
        df: data frame to use
        split: one of train, val, test, all (default: 'train')
        label: class name (default: 'all')
    Returns:
        description (mean, minimum, maximum, 25%, 50%, 75% quantiles) of annotations
    """
    if split != "all":
        df = df[df.split == split].reset_index(drop=True)
    if label != "all":
        if isinstance(label, int):
            df = df[df.label == label].reset_index(drop=True)
        elif len(label) == 6:
            df = df[df.code_s == label].reset_index(drop=True)
        elif len(label) == 4:
            df = df[df.code_m == label].reset_index(drop=True)
        elif len(label) == 2:
            df = df[df.code_l == label].reset_index(drop=True)
    annotations = df[["minX", "minY", "maxX", "maxY"]]
    img = np.zeros((416, 416, 3), np.uint8)
    img = Image.fromarray(img)
    rectangles = np.zeros_like(img)[:, :, 0].astype("float32")
    for i, row in tqdm.tqdm(annotations.iterrows()):
        x1 = int(row["minX"])
        x2 = int(row["maxX"])
        y1 = int(row["minY"])
        y2 = int(row["maxY"])
        rectangles[y1:y2, x1:x2] += 1

    plt.figure(figsize=(10, 6))
    plt.title(f"Annotation Heatmap (split: {split}, label: {label})")
    plt.imshow(img)
    plt.imshow(rectangles, alpha=0.8)
    plt.show()
    return annotations.describe()
