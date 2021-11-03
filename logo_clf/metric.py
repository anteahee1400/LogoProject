import torch
import torch.nn as nn
import torch.nn.functional as F
import torchmetrics


class AccS(nn.Module):
    def __init__(self, k=5, num_classes=392, **kwargs):
        super().__init__()
        self.k = k
        self.num_classes = num_classes

    def forward(self, prob, label):
        if len(label.shape) == 1:
            label = F.one_hot(label, num_classes=self.num_classes)
        sorted_prob = prob.sort(dim=1, descending=True)
        pred = torch.where(
            prob > sorted_prob.values[:, self.k].unsqueeze(1),
            torch.ones_like(label),
            torch.zeros_like(label),
        )
        intersections = [i.sum() for i in (label.long() & pred.long())]
        unions = [i.sum() for i in (label.long() | pred.long())]
        return sum([i / u for i, u in zip(intersections, unions)]) / len(unions)


class AccG(nn.Module):
    def __init__(self, k=5, num_classes=392, **kwargs):
        super().__init__()
        self.k = k
        self.num_classes = num_classes

    def forward(self, prob, label):
        if len(label.shape) == 1:
            label = F.one_hot(label, num_classes=self.num_classes)
        sorted_prob = prob.sort(dim=1, descending=True)
        pred = torch.where(
            prob > sorted_prob.values[:, self.k].unsqueeze(1),
            torch.ones_like(label),
            torch.zeros_like(label),
        )
        correct = sum([i.sum() > 0 for i in (label.long() & pred.long())])
        batch_size = label.shape[0]
        return correct / batch_size


class PrecK(nn.Module):
    def __init__(self, k=5, num_classes=392, **kwargs):
        super().__init__()
        self.k = k
        self.num_classes = num_classes
        self.prec = torchmetrics.Precision(average="samples", mdmc_average="global", ignore_index=0, top_k=k)

    def forward(self, prob, label):
        if len(label.shape) == 1:
            label = F.one_hot(label, num_classes=self.num_classes)
        return self.prec.to(prob.device)(prob, label.long())


class RecK(nn.Module):
    def __init__(self, k=5, num_classes=392, **kwargs):
        super().__init__()
        self.k = k
        self.num_classes = num_classes
        self.rec = torchmetrics.Recall(average="samples", mdmc_average="global", ignore_index=0, top_k=k)

    def forward(self, prob, label):
        if len(label.shape) == 1:
            label = F.one_hot(label, num_classes=self.num_classes)
        return self.rec.to(prob.device)(prob, label.long())


class F1K(nn.Module):
    def __init__(self, k=5, num_classes=392, **kwargs):
        super().__init__()
        self.k = k
        self.num_classes = num_classes
        self.f1 = torchmetrics.F1(average="samples", mdmc_average="global", ignore_index=0, top_k=k)

    def forward(self, prob, label):
        if len(label.shape) == 1:
            label = F.one_hot(label, num_classes=self.num_classes)
        return self.f1.to(prob.device)(prob, label.long())
