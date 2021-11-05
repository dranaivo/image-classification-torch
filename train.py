import os
from operator import itemgetter

import torch
import torch.nn as nn
import torch.optim as optim

from torchvision import datasets, transforms
from torch.optim.lr_scheduler import MultiStepLR

from trainer import Trainer, hooks, configuration
from trainer.data import get_dataloaders
from trainer.models import get_model
from trainer.utils import setup_system, patch_configs
from trainer.metrics import AccuracyEstimator
from trainer.tensorboard_visualizer import TensorBoardVisualizer

#TODO(me): Module docstring (ALL)

class Experiment:
    '''Define the experiment with the given model and given data.'''

    def __init__(self,
                 model: nn.Module, 
                 system_config: configuration.SystemConfig = configuration.
                 SystemConfig(),
                 data_config: configuration.DataConfig = configuration.
                 DataConfig(),
                 optimizer_config: configuration.
                 OptimizerConfig = configuration.OptimizerConfig()):

        self.loader_train, self.loader_test = get_dataloaders(
            data_config.root_dir,
            batch_size=data_config.batch_size,
            num_workers=data_config.num_workers,
            data_augmentation=data_config.data_augmentation)

        setup_system(system_config)

        self.model = model
        self.loss_fn = nn.CrossEntropyLoss()
        # TODO: put choice of topk in cfg.
        self.metric_fn = AccuracyEstimator(topk=(1,))
        # TODO: parametrize the choice of optimizer and scheduler (affect optimizer_config)
        self.optimizer = optim.SGD(self.model.parameters(),
                                   lr=optimizer_config.learning_rate,
                                   weight_decay=optimizer_config.weight_decay,
                                   momentum=optimizer_config.momentum)
        self.lr_scheduler = MultiStepLR(
            self.optimizer,
            milestones=optimizer_config.lr_step_milestones,
            gamma=optimizer_config.lr_multiplier)
        self.visualizer = TensorBoardVisualizer()

    def run(self, trainer_config: configuration.TrainerConfig) -> dict:

        device = torch.device(trainer_config.device)
        self.model = self.model.to(device)
        self.loss_fn = self.loss_fn.to(device)

        model_trainer = Trainer(
            model=self.model,
            loader_train=self.loader_train,
            loader_test=self.loader_test,
            loss_fn=self.loss_fn,
            metric_fn=self.metric_fn,
            optimizer=self.optimizer,
            lr_scheduler=self.lr_scheduler,
            device=device,
            data_getter=itemgetter(0),  #me: a fancy way to extract x (item[0])
            target_getter=itemgetter(1),  #me:a fancy way to extract y (item[1])
            stage_progress=trainer_config.progress_bar,
            get_key_metric=itemgetter("top1"),
            visualizer=self.visualizer,
            model_saving_frequency=trainer_config.model_saving_frequency,
            save_dir=trainer_config.model_dir)

        model_trainer.register_hook("end_epoch",
                                    hooks.end_epoch_hook_classification)
        self.metrics = model_trainer.fit(trainer_config.epoch_num)

        return self.metrics


def main():
    '''Run the experiment
    '''

    data_config = configuration.DataConfig()
    data_config, trainer_config = patch_configs(
        trainer_config=configuration.TrainerConfig,
        data_config=data_config
    )
    #TODO: add model config, for pre-trained
    model = get_model(pretrained=False, n_classes=data_config.n_classes)
    experiment = Experiment(
                            model=model,
                            data_config=data_config)
    results = experiment.run(trainer_config)

    return results


if __name__ == '__main__':
    main()