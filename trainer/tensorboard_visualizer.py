from torch.utils.tensorboard import SummaryWriter
from .visualizer import Visualizer


class TensorBoardVisualizer(Visualizer):
    '''
    TODO: it should be possible to (optionnally) save images, graphs, histograms.
    '''

    def __init__(self, save_dir):
        self._writer = SummaryWriter(save_dir)

    def update_charts(self, train_metric, train_loss, test_metric, test_loss,
                      learning_rate, epoch):
        
        loss_scalars = dict()
        if train_loss is not None:
            loss_scalars["train"] = train_loss
        if test_loss is not None:
            loss_scalars["validation"] = test_loss
        self._writer.add_scalars("loss", loss_scalars, epoch)

        metrics = dict()
        for test_metric_key, test_metric_value in test_metric.items():
            metric_scalars = dict()
            metric_scalars["validation"] = test_metric_value
            metrics[test_metric_key] = metric_scalars

        if train_metric is not None:
            for metric_key, metric_value in train_metric.items():
                metrics[metric_key]["train"] = metric_value
        
        for metric, scalars in metrics.items():
            self._writer.add_scalars(metric, scalars, global_step=epoch)

        self._writer.add_scalar("learning_rate", learning_rate, epoch)

    def close_tensorboard(self):
        self._writer.close()
