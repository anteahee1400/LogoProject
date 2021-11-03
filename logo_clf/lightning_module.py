from autotrainer.model.utils.common import view_x
import torch
import torch.nn.functional as F
import pytorch_lightning as pl
import torchmetrics
from logo_clf.model.efficientnet_pretrained import *
from logo_clf.metric import *


class LogoLightningModule(pl.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self._setup_configure()

    def training_step(self, train_batch, batch_idx):
        x, y = train_batch
        y_ = self.model.forward(x)

        loss = self.calculate_loss(y_, y)
        evaluation = self.calculate_evaluation(y_, y, self.device)

        for key in evaluation.keys():
            self.log(
                f"train_{key}",
                evaluation[key],
                on_step=True,
                on_epoch=True,
                prog_bar=True,
            )

        return {"loss": loss}

    def training_epoch_end(self, outputs):
        avg_loss = torch.stack([out["loss"] for out in outputs]).mean()
        self.log("train_loss", avg_loss)

    def validation_step(self, val_batch, batch_idx):
        x, y = val_batch
        y_ = self.model.forward(x)

        loss = self.calculate_loss(y_, y)
        evaluation = self.calculate_evaluation(y_, y, self.device)

        for key in evaluation.keys():
            self.log(
                f"valid_{key}",
                evaluation[key],
                on_step=True,
                on_epoch=True,
                prog_bar=True,
            )

        return {"val_loss": loss}

    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([out["val_loss"] for out in outputs]).mean()
        self.log("avg_val_loss", avg_loss)

    def test_step(self, test_batch, batch_idx):
        x, y = test_batch
        y_ = self.model.forward(x)

        loss = self.calculate_test_loss(y_, y)
        evaluation = self.calculate_evaluation(y_, y, self.device)

        for key in evaluation.keys():
            self.log(
                f"test_{key}",
                evaluation[key],
                on_step=True,
                on_epoch=True,
                prog_bar=True,
            )

        return {"test_loss": loss}

    def test_epoch_end(self, outputs):
        avg_loss = torch.stack([out["test_loss"] for out in outputs]).mean()
        self.log("avg_test_loss", avg_loss)

    def _setup_configure(self):
        self.model = self.configure_model()
        self.loss_func = self.configure_loss()
        self.test_loss_func = torch.nn.BCEWithLogitsLoss()
        self.eval_funcs = self.configure_evaluation()

    def configure_model(self):
        model_name = self.config["model"]["name"]
        model_kwargs = self.config["model"]["kwargs"]
        model_ckpt = self.config["model"]["ckpt"]
        model = eval(model_name)(**model_kwargs)
        if model_ckpt is not None:
            loaded = torch.load(model_ckpt, map_location="cpu")
            model.load_state_dict(
                {k.replace("model.", ""): v for k, v in loaded["state_dict"].items()}
            )
        return model

    def configure_loss(self):
        loss_name = self.config["loss"]["name"]
        loss_kwargs = self.config["loss"]["kwargs"]
        loss = eval(loss_name)(**loss_kwargs)
        return loss

    def configure_evaluation(self):
        evals = [item["name"] for item in self.config["evaluation"]["items"]]
        evals_kwargs = [item["kwargs"] for item in self.config["evaluation"]["items"]]
        eval_funcs = []
        for func_str, kwargs in zip(evals, evals_kwargs):
            func = eval(func_str)(**kwargs)
            eval_funcs.append(func)
        return eval_funcs

    def configure_optimizers(self):
        optimizer_name = self.config["optimizer"]["name"]
        optimizer_kwargs = self.config["optimizer"]["kwargs"]
        scheduler_name = self.config["scheduler"]["name"]
        scheduler_kwargs = self.config["scheduler"]["kwargs"]
        optimizer = eval(f"torch.optim.{optimizer_name}")(
            self.parameters(), **optimizer_kwargs
        )
        scheduler = eval(f"torch.optim.lr_scheduler.{scheduler_name}")(
            optimizer, **scheduler_kwargs
        )
        return [optimizer], [scheduler]

    def calculate_loss(self, predicted, answer):
        pred = predicted.view(-1, predicted.shape[-1])
        return self.loss_func(pred, answer)

    def calculate_test_loss(self, predicted, answer):
        pred = predicted.view(-1, predicted.shape[-1])
        return self.test_loss_func(pred, answer)

    def calculate_evaluation(self, predicted, answer, device="cpu"):
        evaluations = dict()
        for idx, metric in enumerate(self.eval_funcs):
            key = metric.__class__.__name__.lower() + str(idx)
            pred = predicted.view(-1, predicted.shape[-1])
            prob = F.softmax(pred, dim=1)
            evaluations[key] = metric.to(device)(prob, answer.long())
        return evaluations
