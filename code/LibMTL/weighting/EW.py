import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from LibMTL.weighting.abstract_weighting import AbsWeighting

class EW(AbsWeighting):
    r"""Equal Weighting (EW).

    The loss weight for each task is always ``1 / T`` in every iteration, where ``T`` denotes the number of tasks.

    """
    def __init__(self):
        super(EW, self).__init__()
        # ['creep', 'density', 'liquidus', 'phase_class', 'size', 'solidus', 'solvus']
        # self.task_weights = [10, 1, 1000, 10, 1, 1]
    # def backward(self, losses, **kwargs):
    #     # loss = torch.mul(losses, torch.ones_like(losses).to(self.device)).sum()
    #     loss = torch.mul(losses, torch.ones_like(losses).to(self.device)).sum()
    #     # print(losses)

    #     loss.backward()
    #     return np.ones(self.task_num)

    def backward(self, losses, weights, **kwargs):
        # if self.task_weights is None:
        #     # If task weights are not provided, use equal weighting
        #     task_weights = torch.ones(self.task_num, device=self.device) / self.task_num
        # else:
        #     # Use user-defined task weights
        #     task_weights = torch.tensor(self.task_weights, device=self.device, dtype=torch.float32)

        task_weights = torch.tensor(weights, device=self.device, dtype=torch.float32)
        weighted_losses = torch.mul(losses, task_weights)
        loss = weighted_losses.sum()
        loss.backward()

        # Convert task_weights to NumPy array for return
        return task_weights.cpu().numpy()
