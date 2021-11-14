import os
import pandas as pd
from pathlib import Path
from PIL import Image

import pytorch_lightning as pl
import torch
from torchvision import transforms
import wandb

from logo_clf.utils import read_json
from logo_clf.visualization.gradcam import *


def split_multi_hot(label):
    batch_ids, batch_labels = torch.where(label == 1)
    labels = [[] for _ in range(label.shape[0])]
    for i, idx in enumerate(batch_ids):
        labels[idx].append(batch_labels[i])
    labels = [torch.stack(l) for l in labels]
    max_len = max([len(l) for l in labels])
    return labels, max_len


class WandbCallback(pl.Callback):
    def __init__(self, datamodule, num_samples=32, **kwargs):
        super().__init__()
        datamodule.setup()
        val_samples = next(iter(datamodule.val_dataloader(shuffle=True)))
        test_samples = next(iter(datamodule.test_dataloader(shuffle=True)))
        self.num_samples = num_samples
        self.val_imgs, self.val_labels = val_samples
        self.test_imgs, self.test_labels = test_samples

    def on_keyboard_interrupt(self, trainer, pl_module):
        self.trained_model_artifact(pl_module.model)
        return super().on_keyboard_interrupt(trainer, pl_module)

    def on_train_epoch_start(self, trainer, pl_module):
        if trainer.current_epoch == 0:
            self.init_model_artifact(pl_module.model)

    def init_model_artifact(self, model):
        model_artifact = wandb.Artifact(
            f"{model.__class__.__name__}_init",
            type="model",
            description=f"{model.__class__.__name__} style CNN",
            metadata=dict(wandb.config),
        )
        torch.save(model.state_dict(), "initialized_model.pt")
        model_artifact.add_file("initialized_model.pt")
        wandb.save("initialized_model.pt")
        wandb.log_artifact(model_artifact)

    def trained_model_artifact(self, model):
        model_artifact = wandb.Artifact(
            f"{model.__class__.__name__}_trained",
            type="model",
            description=f"trained {model.__class__.__name__}",
            metadata=dict(wandb.config),
        )
        torch.save(model.state_dict(), "trained_model.pt")
        model_artifact.add_file("trained_model.pt")
        wandb.save("trained_model.pt")
        wandb.log_artifact(model_artifact)


class WandbImageCallback(WandbCallback):
    def on_validation_epoch_end(self, trainer, pl_module):
        val_imgs = self.val_imgs.to(device=pl_module.device)
        val_labels = self.val_labels.to(device=pl_module.device)
        logits = pl_module.model.forward(val_imgs)
        preds = torch.argmax(logits, -1)

        if trainer.current_epoch % 5 == 4:
            self.trained_model_artifact(pl_module.model)

        ## gradcam
        model = pl_module.model
        gradcams, originals = self.gradcams(
            model,
            val_imgs,
            val_labels,
            target_layer="_blocks",
            fc_layer="_fc",
            size=224,
        )

        trainer.logger.experiment.log(
            {
                "originals": originals,
                "gradcams": gradcams,
                "examples": self.predimgs(val_imgs, val_labels, preds),
            },
            commit=False,
        )

    def get_gradcam_img(self, im, cam, size=224):
        img = transforms.ToPILImage()(im)
        npimg = get_class_activation_on_image(img, cam, size=size)
        return get_img(npimg)

    def gradcams(
        self, model, imgs, labels, target_layer="_blocks", fc_layer="_fc", size=224
    ):
        gradcams = []
        originals = []

        torch.set_grad_enabled(True)
        gradcam = GradCam(model, target_layer=target_layer, fc_layer=fc_layer)
        for im, l in zip(imgs, labels):
            if len(l) > 1:
                l = l[0]
            x = im.unsqueeze(0)
            # x = Variable(x.data, requires_grad=True)
            cam = gradcam.generate_cam(x, l.item(), size=size)
            gradcams.append(wandb.Image(self.get_gradcam_img(im, cam, size=size)))
            originals.append(wandb.Image(im))
        return gradcams, originals

    def predimgs(self, imgs, labels, preds):
        return [
            wandb.Image(x, caption=f"Pred:{pred}, Label:{y}")
            for x, pred, y in zip(
                imgs[: self.num_samples],
                preds[: self.num_samples],
                labels[: self.num_samples],
            )
        ]


class LogoImageCallback(WandbImageCallback):
    def __init__(
        self,
        datamodule,
        num_samples=32,
        data_path=None,
        label_col="label",
        label_path=None,
        mapper_path=None,
        k=5,
        **kwargs,
    ):
        super().__init__(datamodule, num_samples=num_samples)
        self.label_col = label_col
        self.meta = self.load_meta(label_path)
        self.mapper = self.load_mapper(mapper_path)
        self.img_mapper = self.load_img_mapper()
        self.k = k
        self.data_path = Path(data_path)
        if self.data_path.name != "train_folder":
            self.data_path = self.data_path / "train_folder"
        self.default_imgpath = "x.jpg"

    def load_mapper(self, path):
        if path is not None:
            return read_json(path)
        else:
            df = self.meta.groupby("label").first().reset_index()
            return {str(row["label"]): row["desc_s"] for i, row in df.iterrows()}

    def load_meta(self, path):
        return pd.read_csv(path, low_memory=False)

    def load_img_mapper(self):
        df = self.meta
        mapper = {}
        for label in df[self.label_col].unique():
            mapper[str(label)] = df[df[self.label_col] == label]["path"].tolist()
        return mapper

    def open_data(self, img_path, rgb=True):
        image = Image.open(img_path)
        if rgb:
            image = image.convert("RGB")
        return image

    def get_imgs_from_mapper(self, labels, num_samples=1):
        all_imgs = []
        for l in labels:
            imgs = self.img_mapper.get(str(l), [])[:num_samples]
            imgs = [self.open_data(os.path.join(self.data_path, im)) for im in imgs]
            all_imgs.append(imgs)
        return all_imgs

    def on_validation_epoch_end(self, trainer, pl_module):
        val_imgs = self.val_imgs.to(device=pl_module.device)
        val_labels = self.val_labels.to(device=pl_module.device)

        logits = pl_module.model.forward(val_imgs)

        if trainer.current_epoch % 5 == 4:
            self.trained_model_artifact(pl_module.model)

        ## gradcam
        gradcams, originals = self.gradcams(
            pl_module.model,
            val_imgs,
            val_labels,
            target_layer="_blocks",
            fc_layer="_fc",
            size=416,
        )

        # table
        test_table = self.log_table(
            val_imgs, val_labels, logits, columns=["image", "label"], k=self.k
        )

        trainer.logger.experiment.log(
            {
                "test_predictions": test_table,
                "originals": originals,
                "gradcams": gradcams,
            },
            commit=False,
        )

    def on_test_epoch_end(self, trainer, pl_module):
        test_imgs = self.test_imgs.to(device=pl_module.device)
        test_labels = self.test_labels.to(device=pl_module.device)
        max_len = 1
        if len(test_labels.shape) == 2:
            test_labels, max_len = split_multi_hot(test_labels)

        logits = pl_module.model.forward(test_imgs)

        if trainer.current_epoch % 5 == 4:
            self.trained_model_artifact(pl_module.model)

        ## gradcam
        gradcams, originals = self.gradcams(
            pl_module.model,
            test_imgs,
            test_labels,
            target_layer="_blocks",
            fc_layer="_fc",
            size=416,
        )

        # table
        columns = ["image"]
        for i in range(max_len):
            columns.append(f"label{i}")
        test_table = self.log_table(
            test_imgs, test_labels, logits, columns=columns, k=self.k
        )

        trainer.logger.experiment.log(
            {
                "test_predictions": test_table,
                "originals": originals,
                "gradcams": gradcams,
            },
            commit=False,
        )

    def log_table(self, real_imgs, labels, outputs, columns=["image", "label"], k=5):
        if isinstance(labels, list):
            labels = [
                [self.mapper.get(str(l), "none") for l in label.cpu().numpy()]
                for label in labels
            ]
        else:
            labels = [self.mapper.get(str(l), "none") for l in labels.cpu().numpy()]
        scores = torch.nn.functional.softmax(outputs, dim=-1)
        batch_k_probs = scores.sort(dim=1, descending=True).values[:, :k].cpu().numpy()
        batch_k_preds = scores.argsort(dim=1, descending=True)[:, :k].cpu().numpy()
        batch_k_predimgs = [self.get_imgs_from_mapper(preds) for preds in batch_k_preds]

        batch_k_wandbimgs = []
        for preds, probs, lists_of_imgs in zip(
            batch_k_preds, batch_k_probs, batch_k_predimgs
        ):
            batch_k_wandbimgs.append(
                [
                    (
                        self.mapper.get(str(l), str(l)),
                        f"{round(p*100, 3)}%",
                        wandb.Image(im[0]),
                    )
                    for l, p, im in zip(preds, probs, lists_of_imgs)
                ]
            )

        col_len = len(columns)
        for i in range(k):
            columns.append(f"pred_{i}_label")
            columns.append(f"pred_{i}_prob")
            columns.append(f"pred_{i}_image")

        all_data = []
        for img, label, limlist in zip(real_imgs, labels, batch_k_wandbimgs):
            if isinstance(label, list):
                data = [wandb.Image(img), *label]
                data.extend(["x"] * (col_len - len(data)))
            else:
                data = [wandb.Image(img), label]
            for k, p, w in limlist:
                data.append(k)
                data.append(p)
                data.append(w)
            all_data.append(data)

        test_table = wandb.Table(
            columns=columns,
            data=all_data,
        )
        return test_table


class LogoObjCallback(LogoImageCallback):
    def on_validation_epoch_end(self, trainer, pl_module):
        val_imgs = self.val_imgs.to(device=pl_module.device)
        val_labels = [
            {k: v.to(device=pl_module.device) for k, v in label.items()}
            for label in self.val_labels
        ]

        logits = pl_module.model.forward(
            val_imgs
        )  # {"pred_logits":torch.Size([32, 100, 84]), "pred_boxes":torch.Size([32, 100, 4])}

        if trainer.current_epoch % 5 == 4:
            self.trained_model_artifact(pl_module.model)

        # bboxes
        test_table = self.log_table(
            val_imgs, val_labels, logits, pl_module.device, p=0.7
        )

        trainer.logger.experiment.log(
            {
                "test_table": test_table,
            },
            commit=False,
        )

    def resize(self, img, size):
        res = transforms.Resize(size)
        return res(img)

    def box_cxcywh_to_xyxy(self, x):
        x_c, y_c, w, h = x.unbind(1)
        b = [(x_c - 0.5 * w), (y_c - 0.5 * h), (x_c + 0.5 * w), (y_c + 0.5 * h)]
        return torch.stack(b, dim=1)

    def rescale_bboxes(self, out_bbox, size, device):
        img_w, img_h = size
        b = self.box_cxcywh_to_xyxy(out_bbox)
        b = b * torch.tensor([img_w, img_h, img_w, img_h], dtype=torch.float32).to(
            device
        )
        return b

    def log_table(self, real_imgs, labels, outputs, device, p=0.7):

        bboxes = []
        for i in range(len(labels)):
            img = real_imgs.tensors[i]
            label = labels[i]
            size = list(label["size"].cpu().numpy())
            pred_logit = outputs["pred_logits"][i]
            pred_box = outputs["pred_boxes"][i]

            bbox = self.bounding_box(
                img, label, pred_logit, pred_box, size, device, p=p
            )
            # test_table.add_data(true_bbox, pred_bbox)
            bboxes.append(bbox)

        return bboxes

    def bounding_box(self, img, label, pred_logit, pred_box, size, device, p=0.7):
        img = self.resize(img, size)
        class_id_to_label = {int(k): v for k, v in self.mapper.items()}
        t_boxes = self.rescale_bboxes(label["boxes"], size, device)
        t_boxes = t_boxes.cpu().numpy()
        all_t_boxes = []
        for b_i, box in enumerate(t_boxes):
            class_id = label["labels"][b_i].item()
            box_data = {
                "position": {
                    "minX": float(box[0]),
                    "maxX": float(box[2]),
                    "minY": float(box[1]),
                    "maxY": float(box[3]),
                },
                "class_id": class_id,
                "box_caption": class_id_to_label.get(class_id, "none"),
                "domain": "pixel",
            }
            all_t_boxes.append(box_data)

        probs = pred_logit.softmax(-1)
        max_probs = probs[:, :-1].max(-1).values
        keep = max_probs > p

        argmaxs = probs.argmax(-1).detach().cpu().numpy()
        p_boxes = self.rescale_bboxes(pred_box, size, device)
        p_boxes = p_boxes.detach().cpu().numpy()
        all_p_boxes = []
        for b_i, (l, box) in enumerate(zip(argmaxs, p_boxes)):
            class_id = int(l)
            if class_id != int(probs.shape[-1]) - 1:
                box_data = {
                    "position": {
                        "minX": float(box[0]),
                        "maxX": float(box[2]),
                        "minY": float(box[1]),
                        "maxY": float(box[3]),
                    },
                    "class_id": class_id,
                    "box_caption": class_id_to_label.get(class_id, str(class_id)),
                    "domain": "pixel",
                    "scores": {"score": max_probs[b_i].item()},
                }
                all_p_boxes.append(box_data)

        box_image = wandb.Image(
            img,
            boxes={
                "predictions": {
                    "box_data": all_p_boxes,
                    "class_labels": class_id_to_label,
                },
                "ground_truth": {
                    "box_data": all_t_boxes,
                    "class_labels": class_id_to_label,
                },
            },
        )
        return box_image
