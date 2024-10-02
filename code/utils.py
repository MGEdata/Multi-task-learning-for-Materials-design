import json
import os
import torch
from torch.utils.data.dataset import Dataset
import random
import numpy as np
import pandas as pd
import torch.nn as nn

def get_embeds(path):
    with open(path, 'r', encoding='utf8') as fe:
        embeds = json.load(fe)
    return embeds

def set_device(gpu_id="0"):
    r"""Set the device where model and data will be allocated.

    Args:
        gpu_id (str, default='0'): The id of gpu.
    """
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu_id

def set_random_seed(seed):
    r"""Set the random seed for reproducibility.

    Args:
        seed (int, default=0): The random seed.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.backends.cudnn.deterministic = True # 将 CuDNN 的随机性设置为确定性模式，即启用确定性算法。
        torch.backends.cudnn.benchmark = False # 禁用了 CuDNN 的自动调整性能的模式。可以确保网络的前向传播在不同输入上也产生相同的结果，这对于一致性测试和验证非常重要。

def json_save(obj, path):
    with open(path,"w", encoding='utf-8') as f:
        f.write(json.dumps(obj, indent=2))



def get_root_dir():
    r"""Return the root path of project."""
    return os.path.dirname(os.path.abspath(__file__))





class AbsMetric(object):
    r"""An abstract class for the performance metrics of a task.

    Attributes:
        record (list): A list of the metric scores in every iteration.
        bs (list): A list of the number of data in every iteration.
    """

    def __init__(self):
        self.record = []
        self.bs = []

    @property
    def update_fun(self, pred, gt):
        r"""Calculate the metric scores in every iteration and update :attr:`record`.

        Args:
            pred (torch.Tensor): The prediction tensor.
            gt (torch.Tensor): The ground-truth tensor.
        """
        pass

    @property
    def score_fun(self):
        r"""Calculate the final score (when an epoch ends).

        Return:
            list: A list of metric scores.
        """
        pass

    def reinit(self):
        r"""Reset :attr:`record` and :attr:`bs` (when an epoch ends).
        """
        self.record = []
        self.bs = []


class RegMetric(AbsMetric):
    def __init__(self):
        super(RegMetric, self).__init__()
        self.abs_record = []
        self.rel_record = []

    def update_fun(self, pred, gt):
        device = pred.device
        # gt = gt.unsqueeze(1)
        # binary_mask = (torch.sum(gt, dim=1) != 0).unsqueeze(1).to(device)
        # pred = pred.masked_select(binary_mask)
        # gt = gt.masked_select(binary_mask)
        abs_err = torch.abs(pred - gt)
        rel_err = torch.abs(pred - gt) / gt
        abs_err = (torch.sum(abs_err) / torch.nonzero(gt, as_tuple=False).size(0)).item()
        rel_err = (torch.sum(rel_err) / torch.nonzero(gt, as_tuple=False).size(0)).item()
        # abs_err = (torch.sum(abs_err) / torch.nonzero(binary_mask, as_tuple=False).size(0)).item()
        # rel_err = (torch.sum(rel_err) / torch.nonzero(binary_mask, as_tuple=False).size(0)).item()
        self.abs_record.append(abs_err)
        self.rel_record.append(rel_err)
        self.bs.append(pred.size()[0])

    def score_fun(self):
        records = np.stack([np.array(self.abs_record), np.array(self.rel_record)])
        batch_size = np.array(self.bs)

        return [(records[i] * batch_size).sum() / (sum(batch_size)) for i in range(2)]

    def reinit(self):
        self.abs_record = []
        self.rel_record = []
        self.bs = []

class AccuracyMetric(AbsMetric):
    r"""Calculate the Mean Absolute Error (MAE).
    """

    def __init__(self):
        super(AccuracyMetric, self).__init__()
        self.accuracy = list()

    def update_fun(self, pred, gt):
        r"""
        """
        # print(pred)
        predicted_labels = (pred >= 0.5).int()
        predicted_labels = torch.flatten(predicted_labels)
        correct = (predicted_labels == gt).sum().item()
        accuracy = correct / (pred.size()[0])
        self.accuracy.append(accuracy) # .item()
        self.bs.append(pred.size()[0])

    def score_fun(self):
        r"""
        """
        records = np.array(self.accuracy)
        batch_size = np.array(self.bs)
        return [(records * batch_size).sum() / (sum(batch_size))]

    def reinit(self):
        self.accuracy = []
        self.bs = []





