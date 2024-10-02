import torch.nn as nn
import numpy as np
import torch

class AbsLoss(object):
    r"""An abstract class for loss functions.
    """

    def __init__(self):
        self.record = []
        self.bs = []

    def compute_loss(self, pred, gt):
        r"""Calculate the loss.

        Args:
            pred (torch.Tensor): The prediction tensor.
            gt (torch.Tensor): The ground-truth tensor.

        Return:
            torch.Tensor: The loss.
        """
        pass

    def _update_loss(self, pred, gt):
        loss = self.compute_loss(pred, gt)
        self.record.append(loss.item())
        self.bs.append(pred.size()[0])
        return loss

    def _average_loss(self):
        record = np.array(self.record)
        bs = np.array(self.bs)
        # print((record * bs).sum() / bs.sum())
        return (record * bs).sum() / bs.sum()

    def _reinit(self):
        self.record = []
        self.bs = []


class BCELoss(AbsLoss):
    r"""The cross-entropy loss function.
    """

    def __init__(self):
        super(BCELoss, self).__init__()
        self.loss_fn = nn.BCELoss()

    def compute_loss(self, pred, gt):
        r"""
        """
        loss = self.loss_fn(pred, gt)
        # print("BCE:",loss)
        return loss


class MSELoss(AbsLoss):
    r"""The Mean Squared Error (MSE) loss function.
    """

    def __init__(self):
        super(MSELoss, self).__init__()

        self.loss_fn = nn.MSELoss()

    def compute_loss(self, pred, gt):
        r"""
        """
        loss = self.loss_fn(pred, gt)
        return loss