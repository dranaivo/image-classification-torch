"""Unified class containing training and evaluation pipeline.

The Trainer class is used inside of a main caller function:

1 - It takes the model and its loss, system, data and training configurations.
2 - It invokes the train and test data loader
3 - Using the data from the loader and the configurations, it is training and evaluating the model.
4 - While training/evaluating, it will send logs to TensorBoard and regularly save the model.
"""

import os

from typing import Union, Callable
from pathlib import Path
from operator import itemgetter

import torch

from tqdm import tqdm
from torch.optim.lr_scheduler import ReduceLROnPlateau

from .hooks import test_hook_default, train_hook_default
from .visualizer import Visualizer


class Trainer:
    """ Generic class for training loop."""

    def __init__(  # pylint: disable=too-many-arguments
            self,
            model: torch.nn.Module,
            loader_train: torch.utils.data.DataLoader,
            loader_test: torch.utils.data.DataLoader,
            loss_fn: Callable,
            metric_fn: Callable,
            optimizer: torch.optim.Optimizer,
            lr_scheduler: Callable,
            device: Union[torch.device, str] = "cuda",
            model_saving_frequency: int = 1,
            save_dir: Union[str, Path] = "checkpoints",
            model_name_prefix: str = "model",
            data_getter: Callable = itemgetter("image"),
            target_getter: Callable = itemgetter("target"),
            stage_progress: bool = True,
            visualizer: Union[Visualizer, None] = None,
            get_key_metric: Callable = itemgetter("top1"),
    ):
        """Constructor.
        
        Args:
            model : nn.Module
                torch model to train
            loader_train : torch.utils.DataLoader
                train dataset loader.
            loader_test : torch.utils.DataLoader
                test dataset loader
            loss_fn : callable
                loss function
            metric_fn : callable
                evaluation metric function
            optimizer : torch.optim.Optimizer
                Optimizer
            lr_scheduler : torch.optim.LrScheduler
                Learning Rate scheduler
            configuration : TrainerConfiguration
                a set of training process parameters
            data_getter : Callable
                function object to extract input data from the sample prepared by dataloader.
            target_getter : Callable
                function object to extract target data from the sample prepared by dataloader.
            visualizer : Visualizer, optional
                shows metrics values (various backends are possible, e.g.: Tensorboard, Matplotlib)
        """
        self.model = model
        self.loader_train = loader_train
        self.loader_test = loader_test
        self.loss_fn = loss_fn
        self.metric_fn = metric_fn
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.device = device
        self.model_saving_frequency = model_saving_frequency
        self.save_dir = save_dir
        self.model_name_prefix = model_name_prefix
        self.stage_progress = stage_progress
        self.data_getter = data_getter
        self.target_getter = target_getter
        self.hooks = {}
        self.visualizer = visualizer
        self.get_key_metric = get_key_metric
        self.metrics = {
            "epoch": [],
            "train_loss": [],
            "test_loss": [],
            "test_metric": []
        }
        self._register_default_hooks()

    def fit(self, epochs):
        """ Fit model method.

        For each epoch:
        1 - Go through the training loop (hooks["train"])
        2 - Go thourgh the evaluation loop  (hooks["test])
        3 - Update visualization and metrics
        4 - Go through the lr scheduler
        5 - Save the model

        Arguments:
            epochs (int): number of epochs to train model.
        """

        iterator = tqdm(range(1, epochs + 1), dynamic_ncols=True)
        for epoch in iterator:

            output_train = self.hooks["train"](
                self.model,
                self.loader_train,
                self.loss_fn,
                self.optimizer,
                self.device,
                prefix="[{}/{}]".format(epoch, epochs),
                stage_progress=self.stage_progress,
                data_getter=self.data_getter,
                target_getter=self.target_getter)

            output_test = self.hooks["test"](self.model,
                                             self.loader_test,
                                             self.loss_fn,
                                             self.metric_fn,
                                             self.device,
                                             prefix="[{}/{}]".format(
                                                 epoch, epochs),
                                             stage_progress=self.stage_progress,
                                             data_getter=self.data_getter,
                                             target_getter=self.target_getter,
                                             get_key_metric=self.get_key_metric)

            if self.visualizer:
                self.visualizer.update_charts(
                    None, output_train['loss'], output_test['metric'],
                    output_test['loss'], self.optimizer.param_groups[0]['lr'],
                    epoch)

            self.metrics['epoch'].append(epoch)
            self.metrics['train_loss'].append(output_train['loss'])
            self.metrics['test_loss'].append(output_test['loss'])
            self.metrics['test_metric'].append(output_test['metric'])

            if self.lr_scheduler is not None:
                if isinstance(self.lr_scheduler, ReduceLROnPlateau):
                    self.lr_scheduler.step(output_train['loss'])
                else:
                    self.lr_scheduler.step()

            if self.hooks["end_epoch"] is not None:
                self.hooks["end_epoch"](iterator, epoch, output_train,
                                        output_test)

            if epoch % self.model_saving_frequency == 0:
                os.makedirs(self.save_dir, exist_ok=True)
                #TODO: include model name prefix (eg: alexnet, resnet) when available
                #TODO: implement save_best
                torch.save(
                    self.model.state_dict(),
                    os.path.join(self.save_dir, "_".join([self.model_name_prefix, "epoch", str(epoch)]) + ".pth")
                )

        return self.metrics

    def register_hook(self, hook_type, hook_fn):
        """ Register hook method.
        
        Args:
            hook_type (string): hook type. For example, wether the function will be used for train or test.
            hook_fn (callable): hook function.
        """
        self.hooks[hook_type] = hook_fn

    def _register_default_hooks(self):
        """It is calling the another method `register_hook` to register training (`train_hook_default`) and validation (`test_hook_default`) 
        loops. `train_hook_default` and `test_hook_default` are defined in the `hook`-module.
        """
        self.register_hook("train", train_hook_default)
        self.register_hook("test", test_hook_default)
        self.register_hook("end_epoch", None)
