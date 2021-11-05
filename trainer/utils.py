'''Implements helper functions.'''

import random
import numpy as np
import torch

from .configuration import SystemConfig, TrainerConfig, DataConfig


class AverageMeter:
    """Streaming metrics.
    
    Computes and stores the average and current value.
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, count=1):
        self.val = val
        self.sum += val * count
        self.count += count
        self.avg = self.sum / self.count

def patch_configs(trainer_config=TrainerConfig,
                  data_config=DataConfig):
    """Standard configurations if GPU is not available.

    Returns:
        The patched trainer_config and data_config.
    """

    if torch.cuda.is_available():
        trainer_config.device = "cuda"
    else:
        trainer_config.device = "cpu"
        data_config.batch_size = 16
        data_config.num_workers = 2

    return data_config, trainer_config


def setup_system(system_config: SystemConfig) -> None:
    '''System setup for reproducibility.'''

    torch.manual_seed(system_config.seed)
    np.random.seed(system_config.seed)
    random.seed(system_config.seed)
    torch.set_printoptions(precision=10)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(system_config.seed)
        torch.backends.cudnn_benchmark_enabled = system_config.cudnn_benchmark_enabled
        torch.backends.cudnn.deterministic = system_config.cudnn_deterministic
