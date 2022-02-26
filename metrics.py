from typing import List
import torch
from torch import nn
import torchmetrics
class WeightedMseLoss(nn.Module):
    """
    Weighted mse loss columnwise
    """

    def __init__(
        self, weights: List[float] = None, reduction: str = "mean",
    ):
        """
        INIT

        :param List[float] weights: List of weights for each joystick
        :param str reduction: "mean" or "sum"
        """

        assert reduction in ["sum", "mean"], (
            f"Reduction method: {reduction} not implemented. "
            f"Available reduction methods: [sum,mean]"
        )

        super(WeightedMseLoss, self).__init__()

        self.reduction = reduction
        if not weights:
            weights = [1.0, 1.0]
        weights = torch.tensor(weights)
        weights.requires_grad = False

        self.register_buffer("weights", weights)

    def forward(self, predicted: torch.tensor, target: torch.tensor,) -> torch.tensor:

        """
        Forward pass

        :param torch.tensor predicted: Predicted values [batch_size, 2]
        :param torch.tensor target: Target values [batch_size, 2]
        :return: Loss [1] if reduction is "mean" else [2]
        """

        if self.reduction == "mean":
            loss_per_joystick: torch.tensor = torch.mean(
                (predicted - target) ** 2, dim=0
            )
            return torch.mean(self.weights * loss_per_joystick)
        else:
            loss_per_joystick: torch.tensor = torch.sum(
                (predicted - target) ** 2, dim=0
            )
            return self.weights * loss_per_joystick


class CrossEntropyLoss(torch.nn.Module):
    """
    Weighted CrossEntropyLoss
    """

    def __init__(
        self,
        weights: List[float] = None,
        reduction: str = "mean",
        label_smoothing: float = 0.0,
    ):
        """
        INIT

        :param List[float] weights: List of weights for each key combination [9]
        :param str reduction: "mean" or "sum"
        :param float label_smoothing: A float in [0.0, 1.0]. Specifies the amount of smoothing when computing the loss
        """

        assert reduction in ["sum", "mean"], (
            f"Reduction method: {reduction} not implemented. "
            f"Available reduction methods: [sum,mean]"
        )

        super(CrossEntropyLoss, self).__init__()

        self.reduction = reduction
        if not weights:
            weights = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]

        weights = torch.tensor(weights)
        weights.requires_grad = False

        self.register_buffer("weights", weights)

        self.CrossEntropyLoss = torch.nn.CrossEntropyLoss(
            reduction=reduction, weight=weights, label_smoothing=label_smoothing,
        )

    def forward(self, predicted: torch.tensor, target: torch.tensor,) -> torch.tensor:

        """
        Forward pass

        :param torch.tensor predicted: Predicted values [batch_size, 9]
        :param torch.tensor target: Target values [batch_size]
        :return: Loss [1] if reduction is "mean" else [9]
        """
        return self.CrossEntropyLoss(predicted.view(-1, 9), target.view(-1).long())


class CrossEntropyLossImageReorder(torch.nn.Module):
    """
    Weighted CrossEntropyLoss for Image Reordering
    """

    def __init__(
        self, label_smoothing: float = 0.0,
    ):
        """
        INIT

        :param float label_smoothing: A float in [0.0, 1.0]. Specifies the amount of smoothing when computing the loss
        """

        super(CrossEntropyLossImageReorder, self).__init__()

        self.CrossEntropyLoss = torch.nn.CrossEntropyLoss(
            label_smoothing=label_smoothing
        )

    def forward(self, predicted: torch.tensor, target: torch.tensor,) -> torch.tensor:

        """
        Forward pass

        :param torch.tensor predicted: Predicted values [batch_size, 5]
        :param torch.tensor target: Target values [batch_size]
        :return: Loss [1]
        """

        return self.CrossEntropyLoss(predicted.view(-1, 5), target.view(-1).long())


class ImageReorderingAccuracy(torchmetrics.Metric):
    """
    Image Reordering Accuracy Metric
    """

    def __init__(self, dist_sync_on_step=False):
        """
        INIT

        :param bool dist_sync_on_step: If True, the metric will be synchronized on step
        """
        super().__init__(dist_sync_on_step=dist_sync_on_step)

        self.add_state("correct", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, preds: torch.Tensor, target: torch.Tensor):
        """
        Update the metric with the given predictions and targets

        :param torch.Tensor preds: Predictions [batch_size, 5]
        :param torch.Tensor target: Target values [batch_size]
        """
        assert (
            preds.size() == target.size()
        ), f"Pred sise: {preds.size()} != Target size: {target.size()}"

        self.correct += torch.sum(torch.all(preds == target, dim=-1))
        self.total += target.size(0)

    def compute(self):
        return self.correct.float() / self.total