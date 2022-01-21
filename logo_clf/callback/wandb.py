import os
import pandas as pd
from pathlib import Path
from PIL import Image

import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from torchvision import transforms
import wandb

from logo_clf.utils import read_json, split_multi_hot
from logo_clf.visualization.gradcam import *


class WandbCallback(pl.Callback):
    def __init__(self, datamodule, **kwargs):
        super().__init__()
        datamodule.setup()
        val_samples = next(iter(datamodule.val_dataloader(shuffle=True)))
        try:
            test_samples = next(iter(datamodule.test_dataloader(shuffle=True)))
        except:
            test_samples = None
        self.val_imgs, self.val_labels = val_samples
        if test_samples is not None:
            self.test_imgs, self.test_labels = test_samples


class WandbImageCallback(WandbCallback):
    def __init__(
        self,
        datamodule,
        gradcam=False,
        target_layer="_blocks",
        fc_layer="_fc",
        **kwargs,
    ):
        super().__init__(datamodule)
        self.gradcam = gradcam
        self.target_layer = target_layer
        self.fc_layer = fc_layer

    def on_validation_epoch_end(self, trainer, pl_module):
        val_imgs = self.val_imgs.to(device=pl_module.device)
        val_labels = self.val_labels.to(device=pl_module.device)
        logits = pl_module.model.forward(val_imgs)
        preds = torch.argmax(logits, -1)

        ## gradcam
        model = pl_module.model
        gradcams, originals = self.gradcams(
            model,
            val_imgs,
            val_labels,
            target_layer=self.target_layer,
            fc_layer=self.fc_layer,
            size=224,
        )

        trainer.logger.experiment.log(
            {
                "originals": originals,
                "gradcams": gradcams,
                "examples": self.predimgs(val_imgs, val_labels, preds),
            }
            if self.gradcam
            else {"examples": self.predimgs(val_imgs, val_labels, preds)},
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
                imgs,
                preds,
                labels,
            )
        ]


class LogoImageCallback(WandbImageCallback):
    def __init__(
        self,
        datamodule,
        gradcam=False,
        target_layer="_blocks",
        fc_layer="_fc",
        data_path=None,
        label_col="label",
        label_path=None,
        mapper_path=None,
        k=5,
        **kwargs,
    ):
        super().__init__(
            datamodule, gradcam=gradcam, target_layer=target_layer, fc_layer=fc_layer
        )
        self.label_col = label_col
        self.meta = self.load_meta(label_path)
        self.mapper = self.load_mapper(mapper_path)
        self.img_mapper = self.load_img_mapper()
        self.k = k
        self.data_path = Path(data_path)
        if self.data_path.name != "train_folder":
            self.data_path = self.data_path / "train_folder"
        self.default_imgpath = "x.jpg"
        self.prob_type = "softmax"

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

    def calc_prob(self, logit):
        if self.prob_type == "sigmoid":
            return torch.sigmoid(logit)
        return F.softmax(logit, dim=-1)

    def on_validation_epoch_end(self, trainer, pl_module):
        val_imgs = self.val_imgs.to(device=pl_module.device)
        val_labels = self.val_labels.to(device=pl_module.device)
        max_len = 1
        if len(val_labels.shape) == 2:
            val_labels, max_len = split_multi_hot(val_labels)
            
        logits = pl_module.model.forward(val_imgs)
        scores = self.calc_prob(logits)
        batch_k_probs = scores.sort(dim=1, descending=True).values[:, :self.k].cpu().numpy()
        batch_k_preds = scores.argsort(dim=1, descending=True)[:, :self.k].cpu().numpy()
        
        ## gradcam
        model = pl_module.model
        if self.gradcam:
            gradcams, originals = self.gradcams(
                model,
                val_imgs,
                val_labels,
                target_layer=self.target_layer,
                fc_layer=self.fc_layer,
                size=416,
            )

        # table
        columns = ["image"]
        for i in range(max_len):
            columns.append(f"label{i}")
        val_table = self.log_table(
            val_imgs, val_labels, batch_k_probs, batch_k_preds, columns=columns
        )

        trainer.logger.experiment.log(
            {
                "val_predictions": val_table,
                "originals": originals,
                "gradcams": gradcams,
            }
            if self.gradcam
            else {"val_predictions": val_table},
            commit=False,
        )

    def on_test_epoch_end(self, trainer, pl_module):
        test_imgs = self.test_imgs.to(device=pl_module.device)
        test_labels = self.test_labels.to(device=pl_module.device)
        max_len = 1
        if len(test_labels.shape) == 2:
            test_labels, max_len = split_multi_hot(test_labels)

        logits = pl_module.model.forward(test_imgs)
        scores = self.calc_prob(logits)
        batch_k_probs = scores.sort(dim=1, descending=True).values[:, :self.k].cpu().numpy()
        batch_k_preds = scores.argsort(dim=1, descending=True)[:, :self.k].cpu().numpy()
        
        ## gradcam
        model = pl_module.model
        if self.gradcam:
            gradcams, originals = self.gradcams(
                model,
                test_imgs,
                test_labels,
                target_layer=self.target_layer,
                fc_layer=self.fc_layer,
                size=416,
            )

        # table
        columns = ["image"]
        for i in range(max_len):
            columns.append(f"label{i}")
        test_table = self.log_table(
            test_imgs, test_labels, batch_k_probs, batch_k_preds, columns=columns
        )

        trainer.logger.experiment.log(
            {
                "test_predictions": test_table,
                "originals": originals,
                "gradcams": gradcams,
            }
            if self.gradcam
            else {"test_predictions": test_table},
            commit=False,
        )

    def log_table(self, real_imgs, labels, probabilities, predictions, columns=["image", "label"]):
        if isinstance(labels, list):
            labels = [
                [self.mapper.get(str(l), "none") for l in label.cpu().numpy()]
                for label in labels
            ]
        else:
            labels = [self.mapper.get(str(l), "none") for l in labels.cpu().numpy()]
        
        predimgs = [self.get_imgs_from_mapper(preds) for preds in predictions]

        wandbimgs = []
        for preds, probs, lists_of_imgs in zip(
            predictions, probabilities, predimgs
        ):
            wandbimgs.append(
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
        for i in range(len(predictions[0])):
            columns.append(f"pred_{i}_label")
            columns.append(f"pred_{i}_prob")
            columns.append(f"pred_{i}_image")

        all_data = []
        for img, label, limlist in zip(real_imgs, labels, wandbimgs):
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


class LogoMultilabelCallback(LogoImageCallback):
    def __init__(
        self,
        datamodule,
        gradcam=False,
        target_layer="_blocks",
        fc_layer="_fc",
        data_path=None,
        label_col="label",
        label_path=None,
        mapper_path=None,
        k=5,
        **kwargs,
    ):
        super().__init__(
            datamodule, 
            gradcam=gradcam, 
            target_layer=target_layer, 
            fc_layer=fc_layer, 
            data_path=data_path,
            label_col=label_col,
            label_path=label_path,
            mapper_path=mapper_path,
            k=k,
        )
        self.prob_type = "sigmoid"
        
    # def on_validation_epoch_end(self, trainer, pl_module):
    #     val_imgs = self.val_imgs.to(device=pl_module.device)
    #     val_labels = self.val_labels.to(device=pl_module.device)

    #     logits = pl_module.model.forward(val_imgs)
    #     scores = torch.sigmoid(logits)
    #     batch_k_probs = scores.sort(dim=1, descending=True).values[:, :self.k].cpu().numpy()
    #     batch_k_preds = scores.argsort(dim=1, descending=True)[:, :self.k].cpu().numpy()
        
    #     ## gradcam
    #     model = pl_module.model
    #     if self.gradcam:
    #         gradcams, originals = self.gradcams(
    #             model,
    #             val_imgs,
    #             val_labels,
    #             target_layer=self.target_layer,
    #             fc_layer=self.fc_layer,
    #             size=416,
    #         )

    #     # table
    #     val_table = self.log_table(
    #         val_imgs, val_labels, batch_k_probs, batch_k_preds, columns=["image", "label"]
    #     )

    #     trainer.logger.experiment.log(
    #         {
    #             "val_predictions": val_table,
    #             "originals": originals,
    #             "gradcams": gradcams,
    #         }
    #         if self.gradcam
    #         else {"val_predictions": val_table},
    #         commit=False,
    #     )

    # def on_test_epoch_end(self, trainer, pl_module):
    #     test_imgs = self.test_imgs.to(device=pl_module.device)
    #     test_labels = self.test_labels.to(device=pl_module.device)
    #     max_len = 1
    #     if len(test_labels.shape) == 2:
    #         test_labels, max_len = split_multi_hot(test_labels)

    #     logits = pl_module.model.forward(test_imgs)
    #     scores = torch.sigmoid(logits)
    #     batch_k_probs = scores.sort(dim=1, descending=True).values[:, :self.k].cpu().numpy()
    #     batch_k_preds = scores.argsort(dim=1, descending=True)[:, :self.k].cpu().numpy()
        
    #     ## gradcam
    #     model = pl_module.model
    #     if self.gradcam:
    #         gradcams, originals = self.gradcams(
    #             model,
    #             test_imgs,
    #             test_labels,
    #             target_layer=self.target_layer,
    #             fc_layer=self.fc_layer,
    #             size=416,
    #         )

    #     # table
    #     columns = ["image"]
    #     for i in range(max_len):
    #         columns.append(f"label{i}")
    #     test_table = self.log_table(
    #         test_imgs, test_labels, batch_k_probs, batch_k_preds, columns=columns
    #     )

    #     trainer.logger.experiment.log(
    #         {
    #             "test_predictions": test_table,
    #             "originals": originals,
    #             "gradcams": gradcams,
    #         }
    #         if self.gradcam
    #         else {"test_predictions": test_table},
    #         commit=False,
    #     )


# class LogoMultilabelCallback(LogoImageCallback):
#     def __init__(
#         self,
#         datamodule,
#         gradcam=False,
#         target_layer="_blocks",
#         fc_layer="_fc",
#         data_path=None,
#         label_col="label",
#         label_path=None,
#         mapper_path=None,
#         k=5,
#         threshold=0.5,
#         **kwargs,
#     ):
#         super().__init__(
#             datamodule,
#             gradcam=gradcam,
#             target_layer=target_layer,
#             fc_layer=fc_layer,
#             data_path=data_path,
#             label_col=label_col,
#             label_path=label_path,
#             mapper_path=mapper_path,
#             k=k,
#         )
#         self.threshold = threshold

#     def log_table(self, real_imgs, labels, probabilities, predictions, columns=["image", "label"], max_len=5):
#         if isinstance(labels, list):
#             labels = [
#                 [self.mapper.get(str(l), "none") for l in label.cpu().numpy()]
#                 for label in labels
#             ]
#         else:
#             labels = [self.mapper.get(str(l), "none") for l in labels.cpu().numpy()]
        
#         predimgs = [self.get_imgs_from_mapper(preds.cpu().numpy()) for preds in predictions]

#         wandbimgs = []
#         for preds, probs, lists_of_imgs in zip(
#             predictions, probabilities, predimgs
#         ):
#             wandbimgs.append(
#                 [
#                     (
#                         self.mapper.get(str(l), str(l)),
#                         f"{round(p*100, 3)}%",
#                         wandb.Image(im[0]),
#                     )
#                     for l, p, im in zip(preds.cpu().numpy(), probs.cpu().numpy(), lists_of_imgs)
#                 ]
#             )

#         col_len = len(columns)
#         for i in range(max_len):
#             columns.append(f"pred_{i}_label")
#             columns.append(f"pred_{i}_prob")
#             columns.append(f"pred_{i}_image")

#         all_data = []
#         for img, label, limlist in zip(real_imgs, labels, wandbimgs):
#             if isinstance(label, list):
#                 data = [wandb.Image(img), *label]
#                 extend_len = max(col_len - len(data), 0)
#                 data.extend(["x"] * extend_len)
#             else:
#                 data = [wandb.Image(img), label]
#             for k, p, w in limlist:
#                 data.append(k)
#                 data.append(p)
#                 data.append(w)          
#             data.extend([None] * (len(columns)-len(data)))
#             all_data.append(data[:len(columns)])
        
#         test_table = wandb.Table(
#             columns=columns,
#             data=all_data,
#         )
#         return test_table
    

#     def on_validation_epoch_end(self, trainer, pl_module):
#         val_imgs = self.val_imgs.to(device=pl_module.device)
#         val_labels = self.val_labels.to(device=pl_module.device)

#         logits = pl_module.model.forward(val_imgs)
#         probs = torch.sigmoid(logits)
#         preds = torch.where(
#             probs > self.threshold,
#             torch.ones_like(val_labels),
#             torch.zeros_like(val_labels),
#         )

#         max_len = 1
#         max_pred_len = 1
#         if len(val_labels.shape) == 2:
#             val_labels, max_len = split_multi_hot(val_labels)
#             preds, max_pred_len = split_multi_hot(preds)
#             probs = [p[pr] for p, pr in zip(probs, preds)]
            
#         ## gradcam
#         model = pl_module.model
#         if self.gradcam:
#             gradcams, originals = self.gradcams(
#                 model,
#                 val_imgs,
#                 val_labels,
#                 target_layer=self.target_layer,
#                 fc_layer=self.fc_layer,
#                 size=416,
#             )
            
#         # table
#         columns = ["image"]
#         for i in range(max_len):
#             columns.append(f"label{i}")
#         val_table = self.log_table(
#             val_imgs, val_labels, probs, preds, columns=columns, max_len=max_pred_len
#         )

#         trainer.logger.experiment.log(
#             {
#                 "val_predictions": val_table,
#                 "originals": originals,
#                 "gradcams": gradcams,
#             }
#             if self.gradcam
#             else {"val_predictions": val_table},
#             commit=False,
#         )

#     def on_test_epoch_end(self, trainer, pl_module):
#         test_imgs = self.test_imgs.to(device=pl_module.device)
#         test_labels = self.test_labels.to(device=pl_module.device)
#         logits = pl_module.model.forward(test_imgs)
#         probs = torch.sigmoid(logits)
#         preds = torch.where(
#             probs > self.threshold,
#             torch.ones_like(test_labels),
#             torch.zeros_like(test_labels),
#         )

#         max_len = 1
#         max_pred_len = 1
#         if len(test_labels.shape) == 2:
#             test_labels, max_len = split_multi_hot(test_labels)
#             preds, max_pred_len = split_multi_hot(preds)
#             probs = [p[pr] for p, pr in zip(probs, preds)]
            
#         ## gradcam
#         model = pl_module.model
#         if self.gradcam:
#             gradcams, originals = self.gradcams(
#                 model,
#                 test_imgs,
#                 test_labels,
#                 target_layer=self.target_layer,
#                 fc_layer=self.fc_layer,
#                 size=416,
#             )

#         # table
#         columns = ["image"]
#         for i in range(max_len):
#             columns.append(f"label{i}")
#         test_table = self.log_table(
#             test_imgs, test_labels, probs, preds, columns=columns, max_len=5
#         )

#         trainer.logger.experiment.log(
#             {
#                 "test_predictions": test_table,
#                 "originals": originals,
#                 "gradcams": gradcams,
#             }
#             if self.gradcam
#             else {"test_predictions": test_table},
#             commit=False,
#         )

