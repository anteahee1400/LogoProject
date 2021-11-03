import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import tqdm

from PIL import Image


def code_to_str(df):
    if df["code_s"].dtype.name != "object":
        df["code_s"] = df["code_s"].apply(lambda x: str(x).zfill(6))
    if df["code_m"].dtype.name != "object":
        df["code_m"] = df["code_m"].apply(lambda x: str(x).zfill(4))
    if df["code_l"].dtype.name != "object":
        df["code_l"] = df["code_l"].apply(lambda x: str(x).zfill(2))
    return df


def mapper(df, key="code_s", value="desc_s"):
    df = code_to_str(df)
    df = df.groupby(key).first().reset_index()
    return {row[key]: row[value] for i, row in df.iterrows()}


def bar(df, title, xlabel="index", ylabel="count"):
    plt.figure(figsize=(10, 6))
    plt.title(title)
    if xlabel == "index":
        plt.bar(df.index, df[ylabel])
    else:
        plt.bar(df[xlabel], df[ylabel])
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.show()


def class_balance(df, class_name="code_s", visual=False):
    mapping_dict = mapper(df, key=class_name)
    df = code_to_str(df)
    df = (
        df.groupby(class_name)
        .count()
        .reset_index()
        .sort_values("path", ascending=False)
        .reset_index(drop=True)
    )
    df = df[[class_name, "path"]].rename(columns={"path": "count"})
    df["desc"] = df[class_name].apply(lambda x: mapping_dict[x])
    if visual:
        bar(df, "Class Distribution")
    return df

def number_of_classes_by_image(df, visual=False):
    df = df.groupby('path').count().reset_index().sort_values('label', ascending=False).reset_index(drop=True)
    df = df[['path', 'label']].rename(columns={'path':'count', 'label':'num_classes'})
    df = df.groupby('num_classes').count().reset_index().sort_values('count', ascending=False).reset_index(drop=True)
    df = df[['num_classes', 'count']]
    df['percentage'] = df['count'] * 100 / df['count'].sum()
    df['percentage'] = df['percentage'].apply(lambda x: f"{x:.02f}%")
    if visual:
        bar(df.head(10), "Number of Classes by Image", xlabel="num_classes")
    return df

def average_width_and_height(df):
    return np.mean(df.width), np.mean(df.height)


def number_of_images(df):
    return len(df.path.unique())


def median_image_ratio(df):
    df = df[['width', 'height']].describe()
    return df.loc['50%'].width, df.loc['50%'].height


def dimension_insights():
    pass


def annotation_heatmap(df, split='train', label='all'):
    if split != 'all':
        df = df[df.split==split].reset_index(drop=True)
    if label != 'all':
        if isinstance(label, int):
            df = df[df.label==label].reset_index(drop=True)
        elif len(label)==6:
            df = df[df.code_s==label].reset_index(drop=True)
        elif len(label)==4:
            df = df[df.code_m==label].reset_index(drop=True)
        elif len(label)==2:
            df = df[df.code_l==label].reset_index(drop=True)
    annotations = df[['minX', 'minY', 'maxX', 'maxY']]
    img  = np.zeros((416, 416, 3), np.uint8)
    img = Image.fromarray(img)
    rectangles = np.zeros_like(img)[:,:,0].astype('float32')
    for i, row in tqdm.tqdm(annotations.iterrows()):
        x1 = int(row['minX'])
        x2 = int(row['maxX'])
        y1 = int(row['minY'])
        y2 = int(row['maxY'])
        rectangles[y1:y2, x1:x2]+=1
        
    
    plt.figure(figsize=(10, 6))
    plt.title(f"Annotation Heatmap (split: {split}, label: {label})")
    plt.imshow(img)
    plt.imshow(rectangles, alpha=0.8)
    plt.show()
    return annotations.describe()

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
