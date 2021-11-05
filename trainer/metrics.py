'''
Evalution metrics. All metric classes inherit from a base metric class.
'''

from abc import ABC, abstractmethod
import torch
from .utils import AverageMeter


class BaseMetric(ABC):

    @abstractmethod
    def update_value(self, output, target):
        pass

    @abstractmethod
    def get_metric_value(self):
        pass

    @abstractmethod
    def reset(self):
        pass


class AccuracyEstimator(BaseMetric):
    """Computes the precision@k.
    
    A prediction is considered correct if the groundtruth class is among
    the top k classes (class prob. ranking).

    Attributes:
        topk (Tuple) : Each element is an int, representing k.
    """

    def __init__(self, topk=(1,)):
        self.topk = topk
        self.metrics = [AverageMeter() for i in range(len(topk) + 1)]

    def reset(self):
        for i in range(len(self.metrics)):
            self.metrics[i].reset()

    def update_value(self, pred, target):
        with torch.no_grad():
            maxk = max(self.topk)
            batch_size = target.size(0)

            _, pred = pred.topk(maxk, 1, True, True)
            pred = pred.t()
            correct = pred.eq(target.view(1, -1).expand_as(pred))

            for i, k in enumerate(self.topk):
                correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
                self.metrics[i].update(
                    correct_k.mul_(100.0 / batch_size).item())

    def get_metric_value(self):
        metrics = {}
        for i, k in enumerate(self.topk):
            metrics["top{}".format(k)] = self.metrics[i].avg
        return metrics
