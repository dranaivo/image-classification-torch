'''Configurations.

All configurations are decorated with the python @dataclass.
Their attributes can be overriden, for example by user-supplied arguments. 
'''

from typing import Callable, Iterable
from dataclasses import dataclass
from torchvision.transforms import ToTensor

#TODO: configuration for Visualizer 

@dataclass
class SystemConfig:
    '''System Configuration.
    
    Members:
        seed: seed number to set the state of all random number generators.
        cudnn_benchmark_enabled: enable CuDNN benchmark for the sake of performance
        cudnn_deterministic: make cudnn deterministic (reproducible training)
    '''
    seed: int = 42
    cudnn_benchmark_enabled: bool = False
    cudnn_deterministic: bool = True


@dataclass
class DataConfig:
    '''Data Configuration.

    For the dataset and dataloader.

    Members:
        root_dir: dataset directory root.
        batch_size: number of images in each batch.
        num_workers: number of concurrent processes used to prepare data.
    '''
    root_dir: str = "/home/dranaivo/Datasets/Object_classification_2D/cat-dog-panda"
    n_classes: int = 3
    data_augmentation: bool = False
    batch_size: int = 2
    num_workers: int = 2


@dataclass
class OptimizerConfig:
    '''Optimizer Configuration.
    
    Configuration for the torch.optim.SGD optimizer (SGD with momentum). 

    Members:
        learning_rate: determines the speed of network's weights update.
        momentum: used to improve vanilla SGD algorithm and provide better handling of local minimas.
        weight_decay: amount of additional regularization on the weights values.
        lr_step_milestones: when using a scheduler, at which epoches should we 
            make a "step" in learning rate (i.e. decrease it in some manner)
        lr_multiplier: multiplier applied to current learning rate at each of lr_step_milestones
    '''
    learning_rate: float = 1e-3
    momentum: float = 0.9
    weight_decay: float = 0.0001
    lr_step_milestones: Iterable = (30, 40)
    lr_multiplier: float = 0.1


@dataclass
class TrainerConfig:
    '''Training Configuration

    Members:
        model_dir: directory to save model states
        visualizer_dir: directory to save visualizer logs (i.e. Tensorboard)
        model_saving_frequency: frequency of model state savings per epochs
        device: device to use for training.
        epoch_num: number of times the whole dataset will be passed through the network.
        progress_bar: enable progress bar visualization during train process.
    '''
    model_dir: str = "checkpoints/alexnet/"
    visualizer_dir: str = "logs/"
    model_saving_frequency: int = 10
    device: str = "gpu"
    epoch_num: int = 50
    progress_bar: bool = True
