import os
import time
from PIL import Image
import pandas as pd
import io
import torch
import torch.nn.functional as F
from torchvision import transforms
import streamlit as st

from logo_clf.model.efficientnet_pretrained import *
from logo_clf.model.vision_transformer import *
from logo_clf.utils import read_json


def calc_prob(logit, prob_type="softmax"):
    if prob_type == "sigmoid":
        return torch.sigmoid(logit)
    return F.softmax(logit, dim=-1)


def transform(img, size=(416, 416)):
    tran = transforms.Compose(
        [
            transforms.Resize(size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    return tran(img)


def model_load(ckpt, num_classes=392):
    model = efficientnet_b5_pretrained(num_classes=num_classes)
    loaded = torch.load(ckpt, map_location="cpu")
    model.load_state_dict(
        {k.replace("model.", ""): v for k, v in loaded["state_dict"].items()}
    )
    return model


def main():
    st.set_page_config(layout="wide")

    st.markdown(
        '<h1 style="text-align: center; color: white;">LOGO Vienna code classification</h1>',
        unsafe_allow_html=True,
    )

    desc_mapper = read_json("data/label_s_to_desc_s_ko.json")
    code_mapper = read_json("data/label_s_to_code_s.json")
    k = st.sidebar.number_input("top K prediction", value=5)
    device = st.sidebar.selectbox("device", options=["cuda:0", "cuda:1", "cpu"])
    ckpt_dir = st.sidebar.selectbox("ckpt", options=os.listdir("ckpt"))
    ckpt_dir = os.path.join("ckpt", ckpt_dir)
    last_ckpt_file = os.listdir(ckpt_dir)[-1]
    ckpt = os.path.join(ckpt_dir, last_ckpt_file)
    
    st.sidebar.header("Vienna code")
    df = pd.DataFrame(
        {"label": list(desc_mapper.keys()), "desc": list(desc_mapper.values())}
    )
    st.sidebar.dataframe(df)

    image = st.file_uploader("Upload number image", type=["jpg", "jpeg", "png"])
    run = st.button("RUN")
    col1, col2 = st.columns(2)

    if image is not None:

        image = image.read()
        image = Image.open(io.BytesIO(image)).convert("RGB")
        col1.image(image, width=400)

    if run:
        device = torch.device(device)
        model = model_load(ckpt)
        model.eval()
        model.to(device)
        if image is not None:
            input_image = transform(image).to(device)
            start = time.time()
            prediction = model(input_image.unsqueeze(0))
            end = time.time()
            inference_time = end - start
            for prob_type in ["softmax", "sigmoid"]:
                probability = calc_prob(prediction, prob_type=prob_type)

                probs = (
                    probability.sort(dim=1, descending=True)
                    .values.cpu()
                    .detach()
                    .numpy()[0]
                    .tolist()[:k]
                )
                indices = (
                    probability.sort(dim=1, descending=True)
                    .indices.cpu()
                    .detach()
                    .numpy()[0]
                    .tolist()[:k]
                )
                description = [desc_mapper[str(i)] for i in indices]
                codes = [code_mapper[str(i)] for i in indices]
                df = pd.DataFrame(
                    {
                        "label": indices,
                        "code": codes,
                        "description": description,
                        "prob": probs,
                    }
                )

                if prob_type == "softmax":
                    col2.text("Top K prediction")
                else:
                    col2.text("Precise prediction")
                    df = df[df.prob > 0.5]

                df["prob"] = df["prob"].apply(lambda x: f"{round(x*100, 3)}%")
                col2.dataframe(df)
        st.text(f"inference time: {round(inference_time, 3)}s")


if __name__ == "__main__":
    main()
