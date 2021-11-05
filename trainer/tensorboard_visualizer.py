from torch.utils.tensorboard import SummaryWriter
from .visualizer import Visualizer


class TensorBoardVisualizer(Visualizer):
    '''
    TODO: it should be possible to (optionnally) save images, graphs, histograms.
    TODO: use .add_scalars to group train/test loss and train/test metrics together on the same graph
    '''

    def __init__(self, save_dir):
        self._writer = SummaryWriter(save_dir)

    def update_charts(self, train_metric, train_loss, test_metric, test_loss,
                      learning_rate, epoch):
        '''Update all boards'''
        if train_metric is not None:
            for metric_key, metric_value in train_metric.items():
                self._writer.add_scalar(
                    "data/train_metric:{}".format(metric_key), metric_value,
                    epoch)

        for test_metric_key, test_metric_value in test_metric.items():
            self._writer.add_scalar(
                "data/test_metric:{}".format(test_metric_key),
                test_metric_value, epoch)

        if train_loss is not None:
            self._writer.add_scalar("data/train_loss", train_loss, epoch)
        if test_loss is not None:
            self._writer.add_scalar("data/test_loss", test_loss, epoch)

        self._writer.add_scalar("data/learning_rate", learning_rate, epoch)

    def close_tensorboard(self):
        self._writer.close()
