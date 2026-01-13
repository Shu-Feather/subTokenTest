import torch
from torch import nn 
import torch.nn.functional as F
class CrossEntropyLossWithPositionalWeights(nn.CrossEntropyLoss):
    def __init__(self, weight=None, size_average=None, ignore_index=-100, reduce=None, reduction='mean'):
        self.orig_loss = (weight is None or len(weight.shape) == 1)
        if self.orig_loss:
            super().__init__(weight=weight, size_average=size_average, ignore_index=ignore_index, reduce=reduce, reduction=reduction)
        else:
            super().__init__(weight=None, size_average=size_average, ignore_index=ignore_index, reduce=reduce, reduction="none")
        self._pos_weight = weight
        self._reduction = reduction
    
    def forward(self, input, target):
        """
        input: torch.Tensor, shape=(batch_size, num_classes, d1, d2, ...)
            The input tensor.
        target: torch.Tensor, shape=(batch_size, d1, d2, ...)
            The target tensor.
        weights: torch.Tensor, shape=(num_classes, d1, d2, ...)
            The weights tensor.
        ------
        return: torch.Tensor
            The loss value.
        """ 
        if self.orig_loss:
            return super().forward(input, target)

        assert self._pos_weight is not None
        targets_one_hot = F.one_hot(target, num_classes=input.shape[1]).permute(0, -1, *range(1, len(target.shape))).float() # (N, C, d1, d2, ...)
        log_prob = F.log_softmax(input, dim=1) # (N, C, d1, d2, ...)
        loss = - torch.sum(log_prob * targets_one_hot * self._pos_weight.unsqueeze(0), dim = 1) # (N, d1, d2, ...)
        if self._reduction == "mean":
            return torch.mean(loss) # scalar
        elif self._reduction == "sum":
            return torch.sum(loss)
        elif self._reduction == "none":
            return loss
        else:
            raise ValueError(f"Unknown reduction: {self._reduction}")
        