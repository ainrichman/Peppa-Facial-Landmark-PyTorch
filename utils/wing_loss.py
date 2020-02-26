import torch
import math
from torch.nn.modules.loss import _Loss


class WingLoss(_Loss):
    def __init__(self, width=10, curvature=2.0, reduction="mean"):
        super(WingLoss, self).__init__(reduction=reduction)
        self.width = width
        self.curvature = curvature

    def forward(self, prediction, target):
        return self.wing_loss(prediction, target, self.width, self.curvature, self.reduction)

    def wing_loss(self, prediction, target, width=10, curvature=2.0, reduction="mean"):
        diff_abs = (target - prediction).abs()
        loss = diff_abs.clone()
        idx_smaller = diff_abs < width
        idx_bigger = diff_abs >= width
        # loss[idx_smaller] = width * torch.log(1 + diff_abs[idx_smaller] / curvature)
        loss_smaller = width * torch.log(1 + diff_abs[idx_smaller] / curvature)
        C = width - width * math.log(1 + width / curvature)
        # loss[idx_bigger] = loss[idx_bigger] - C
        loss_biger = loss[idx_bigger] - C
        loss = torch.cat((loss_smaller, loss_biger), 0)
        if reduction == "sum":
            loss = loss.sum()
        if reduction == "mean":
            loss = loss.mean()
        return loss
