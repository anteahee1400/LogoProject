import torch
import torch.nn as nn
import torch.nn.functional as F
import torchmetrics
from logo_clf.utils import read_json, split_multi_hot, concat_multi_hot

# num_classes
# code l : 25
# code m : 83
# code s : 392

LABEL_S_TO_CODE_L = read_json("data/label_s_to_code_l.json")
LABEL_S_TO_CODE_M = read_json("data/label_s_to_code_m.json")
CODE_L_TO_LABEL_L = read_json("data/code_l_to_label_l.json")
CODE_M_TO_LABEL_M = read_json("data/code_m_to_label_m.json")

def change_label_to_code_type(label_unstack, label_to_code, code_to_label):
    label_unstack_new = []
    for label in label_unstack:
        label_new = torch.tensor([code_to_label[label_to_code[str(l.item())]] for l in label], dtype=label.dtype).to(label.device)
        label_unstack_new.append(label_new)
    return label_unstack_new


class Acc(nn.Module):
    def __init__(self, k=5, num_classes=392, name="acc", **kwargs):
        super().__init__()
        self.name = name
        self.k = k
        self.num_classes = num_classes
        self.label_to_code = None
        self.code_to_label = None
        if "_l" in name:
            self.num_classes = 25
            self.label_to_code = LABEL_S_TO_CODE_L
            self.code_to_label = CODE_L_TO_LABEL_L
        elif "_m" in name:
            self.num_classes = 83
            self.label_to_code = LABEL_S_TO_CODE_M
            self.code_to_label = CODE_M_TO_LABEL_M
        self.acc = torchmetrics.Accuracy()
        
    def forward(self, prob, label):
        pred = torch.argmax(prob, dim=-1)
        if self.label_to_code is not None:
            label = torch.tensor([self.code_to_label[self.label_to_code[l.item()]] for l in label]).to(prob.device)
            pred = torch.tensor([self.code_to_label[self.label_to_code[p.item()]] for p in pred]).to(prob.device)
        return self.acc.to(prob.device)(pred.long, label.long)


class AccS(nn.Module):
    def __init__(self, k=5, num_classes=392, name="accg", **kwargs):
        super().__init__()
        self.name = name
        self.k = k
        self.num_classes = num_classes
        self.label_to_code = None
        self.code_to_label = None
        if "_l" in name:
            self.num_classes = 25
            self.label_to_code = LABEL_S_TO_CODE_L
            self.code_to_label = CODE_L_TO_LABEL_L
        elif "_m" in name:
            self.num_classes = 83
            self.label_to_code = LABEL_S_TO_CODE_M
            self.code_to_label = CODE_M_TO_LABEL_M

    def forward(self, prob, label):
        if len(label.shape) == 1:
            label = F.one_hot(label, num_classes=392)
        sorted_prob = prob.sort(dim=1, descending=True)
        pred = torch.where(
            prob > sorted_prob.values[:, self.k].unsqueeze(1),
            torch.ones_like(label),
            torch.zeros_like(label),
        )
        
        if self.label_to_code is not None:
            label_unstack, _ = split_multi_hot(label)
            label_unstack_new = change_label_to_code_type(label_unstack, label_to_code=self.label_to_code, code_to_label=self.code_to_label)
            label = concat_multi_hot(label_unstack_new, num_classes=self.num_classes)
            
            pred_unstack, _ = split_multi_hot(pred)
            pred_unstack_new = change_label_to_code_type(pred_unstack, label_to_code=self.label_to_code, code_to_label=self.code_to_label)
            pred = concat_multi_hot(pred_unstack_new, num_classes=self.num_classes)
        
        intersections = [i.sum() for i in (label.long() & pred.long())]
        unions = [i.sum() for i in (label.long() | pred.long())]
        return sum([i / u for i, u in zip(intersections, unions)]) / len(unions)


class AccG(AccS):
    
    def forward(self, prob, label):
        if len(label.shape) == 1:
            label = F.one_hot(label, num_classes=392)
        sorted_prob = prob.sort(dim=1, descending=True)
        pred = torch.where(
            prob > sorted_prob.values[:, self.k].unsqueeze(1),
            torch.ones_like(label),
            torch.zeros_like(label),
        )
        
        if self.label_to_code is not None:
            label_unstack, _ = split_multi_hot(label)
            label_unstack_new = change_label_to_code_type(label_unstack, label_to_code=self.label_to_code, code_to_label=self.code_to_label)
            label = concat_multi_hot(label_unstack_new, num_classes=self.num_classes)
            
            pred_unstack, _ = split_multi_hot(pred)
            pred_unstack_new = change_label_to_code_type(pred_unstack, label_to_code=self.label_to_code, code_to_label=self.code_to_label)
            pred = concat_multi_hot(pred_unstack_new, num_classes=self.num_classes)
            
        correct = sum([i.sum() > 0 for i in (label.long() & pred.long())])
        batch_size = label.shape[0]
        return correct / batch_size


class PrecK(AccS):
    # def __init__(self, k=5, num_classes=392, name="preck", **kwargs):
    #     super().__init__(k=k, num_classes=num_classes, name=name)
        
    #     self.prec = torchmetrics.Precision(
    #         average="samples", mdmc_average="global"
    #     )

    def forward(self, prob, label):
        if len(label.shape) == 1:
            label = F.one_hot(label, num_classes=392)
        sorted_prob = prob.sort(dim=1, descending=True)
        pred = torch.where(
            prob > sorted_prob.values[:, self.k].unsqueeze(1),
            torch.ones_like(label),
            torch.zeros_like(label),
        )
        if self.label_to_code is not None:
            label_unstack, _ = split_multi_hot(label)
            label_unstack_new = change_label_to_code_type(label_unstack, label_to_code=self.label_to_code, code_to_label=self.code_to_label)
            label = concat_multi_hot(label_unstack_new, num_classes=self.num_classes)
            
            pred_unstack, _ = split_multi_hot(pred)
            pred_unstack_new = change_label_to_code_type(pred_unstack, label_to_code=self.label_to_code, code_to_label=self.code_to_label)
            pred = concat_multi_hot(pred_unstack_new, num_classes=self.num_classes)
        
        batch_correct = sum([i.sum() for i in (label.long() & pred.long())])
        return batch_correct / pred.long().sum()



class RecK(AccS):
    # def __init__(self, k=5, num_classes=392, name="reck", **kwargs):
    #     super().__init__(k=k, num_classes=num_classes, name=name)
    #     self.rec = torchmetrics.Recall(
    #         average="samples", mdmc_average="global"
    #     )

    def forward(self, prob, label):
        if len(label.shape) == 1:
            label = F.one_hot(label, num_classes=392)
        sorted_prob = prob.sort(dim=1, descending=True)
        pred = torch.where(
            prob > sorted_prob.values[:, self.k].unsqueeze(1),
            torch.ones_like(label),
            torch.zeros_like(label),
        )
        if self.label_to_code is not None:
            label_unstack, _ = split_multi_hot(label)
            label_unstack_new = change_label_to_code_type(label_unstack, label_to_code=self.label_to_code, code_to_label=self.code_to_label)
            label = concat_multi_hot(label_unstack_new, num_classes=self.num_classes)
            
            pred_unstack, _ = split_multi_hot(pred)
            pred_unstack_new = change_label_to_code_type(pred_unstack, label_to_code=self.label_to_code, code_to_label=self.code_to_label)
            pred = concat_multi_hot(pred_unstack_new, num_classes=self.num_classes)
            
        batch_correct = sum([i.sum() for i in (label.long() & pred.long())])
        return batch_correct / label.long().sum()


class F1K(AccS):
    # def __init__(self, k=5, num_classes=392, name="f1k", **kwargs):
    #     super().__init__(k=k, num_classes=num_classes, name=name)
    #     self.f1 = torchmetrics.F1(average="samples", mdmc_average="global")

    def forward(self, prob, label):
        if len(label.shape) == 1:
            label = F.one_hot(label, num_classes=392)
        sorted_prob = prob.sort(dim=1, descending=True)
        pred = torch.where(
            prob > sorted_prob.values[:, self.k].unsqueeze(1),
            torch.ones_like(label),
            torch.zeros_like(label),
        )
        if self.label_to_code is not None:
            label_unstack, _ = split_multi_hot(label)
            label_unstack_new = change_label_to_code_type(label_unstack, label_to_code=self.label_to_code, code_to_label=self.code_to_label)
            label = concat_multi_hot(label_unstack_new, num_classes=self.num_classes)
            
            pred_unstack, _ = split_multi_hot(pred)
            pred_unstack_new = change_label_to_code_type(pred_unstack, label_to_code=self.label_to_code, code_to_label=self.code_to_label)
            pred = concat_multi_hot(pred_unstack_new, num_classes=self.num_classes)
            
        batch_correct = sum([i.sum() for i in (label.long() & pred.long())])
        prec = batch_correct / pred.long().sum()
        rec = batch_correct / label.long().sum()
        if prec + rec == 0:
            return 0
        f1 = 2*prec*rec / (prec + rec)
        return f1


class Precision(nn.Module):
    def __init__(self, k=5, num_classes=392, name="prec", threshold=0.5, **kwargs):
        super().__init__()
        self.name = name
        self.k = k
        self.num_classes = num_classes
        self.threshold = threshold
        self.label_to_code = None
        self.code_to_label = None
        if "_l" in name:
            self.num_classes = 25
            self.label_to_code = LABEL_S_TO_CODE_L
            self.code_to_label = CODE_L_TO_LABEL_L
        elif "_m" in name:
            self.num_classes = 83
            self.label_to_code = LABEL_S_TO_CODE_M
            self.code_to_label = CODE_M_TO_LABEL_M

    def forward(self, prob, label):
        if len(label.shape) == 1:
            label = F.one_hot(label, num_classes=392)
        pred = torch.where(
            prob > self.threshold,
            torch.ones_like(label),
            torch.zeros_like(label),
        )
        
        if self.label_to_code is not None:
            label_unstack, _ = split_multi_hot(label)
            label_unstack_new = change_label_to_code_type(label_unstack, label_to_code=self.label_to_code, code_to_label=self.code_to_label)
            label = concat_multi_hot(label_unstack_new, num_classes=self.num_classes)
            
            pred_unstack, _ = split_multi_hot(pred)
            pred_unstack_new = change_label_to_code_type(pred_unstack, label_to_code=self.label_to_code, code_to_label=self.code_to_label)
            pred = concat_multi_hot(pred_unstack_new, num_classes=self.num_classes)
        
        batch_correct = sum([i.sum() for i in (label.long() & pred.long())])
        return batch_correct / pred.long().sum()
    

class Recall(Precision):

    def forward(self, prob, label):
        if len(label.shape) == 1:
            label = F.one_hot(label, num_classes=392)
        pred = torch.where(
            prob > self.threshold,
            torch.ones_like(label),
            torch.zeros_like(label),
        )
        
        if self.label_to_code is not None:
            label_unstack, _ = split_multi_hot(label)
            label_unstack_new = change_label_to_code_type(label_unstack, label_to_code=self.label_to_code, code_to_label=self.code_to_label)
            label = concat_multi_hot(label_unstack_new, num_classes=self.num_classes)
            
            pred_unstack, _ = split_multi_hot(pred)
            pred_unstack_new = change_label_to_code_type(pred_unstack, label_to_code=self.label_to_code, code_to_label=self.code_to_label)
            pred = concat_multi_hot(pred_unstack_new, num_classes=self.num_classes)
        
        batch_correct = sum([i.sum() for i in (label.long() & pred.long())])
        return batch_correct / label.long().sum()
    

class F1(Precision):
    
    def forward(self, prob, label):
        if len(label.shape) == 1:
            label = F.one_hot(label, num_classes=392)
        pred = torch.where(
            prob > self.threshold,
            torch.ones_like(label),
            torch.zeros_like(label),
        )
        
        if self.label_to_code is not None:
            label_unstack, _ = split_multi_hot(label)
            label_unstack_new = change_label_to_code_type(label_unstack, label_to_code=self.label_to_code, code_to_label=self.code_to_label)
            label = concat_multi_hot(label_unstack_new, num_classes=self.num_classes)
            
            pred_unstack, _ = split_multi_hot(pred)
            pred_unstack_new = change_label_to_code_type(pred_unstack, label_to_code=self.label_to_code, code_to_label=self.code_to_label)
            pred = concat_multi_hot(pred_unstack_new, num_classes=self.num_classes)
        
        batch_correct = sum([i.sum() for i in (label.long() & pred.long())])
        prec = batch_correct / pred.long().sum()
        rec = batch_correct / label.long().sum()
        if prec + rec == 0:
            return 0
        f1 = 2*prec*rec / (prec + rec)
        return f1