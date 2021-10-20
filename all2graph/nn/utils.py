import numpy as np
import torch

from toad.nn import Module
from ..parsers import RawGraphParser
from ..data import DataLoader
from ..version import __version__
from ..utils import progress_wrapper


def num_parameters(module: torch.nn.Module):
    parameters = {}
    for p in module.parameters():
        parameters[id(p)] = p
    return sum(map(lambda x: np.prod(x.shape), parameters.values()))


class MyModule(Module):
    def __init__(self, raw_graph_parser: RawGraphParser):
        super().__init__()
        self.version = __version__
        self.raw_graph_parser = raw_graph_parser
        self._loss = None
        self._optimizer = None

    def predict_dataloader(self, dataloader: DataLoader, postfix=''):
        with torch.no_grad():
            self.eval()
            labels = None
            outputs = None
            for graph, label in progress_wrapper(dataloader, postfix=postfix):
                output = self(graph)
                if outputs is None:
                    outputs = {k: [v.detach().cpu()] for k, v in output.items()}
                    labels = {k: [v.detach().cpu()] for k, v in label.items()}
                else:
                    for k in output:
                        outputs[k].append(output[k].detach().cpu())
                        labels[k].append(label[k].detach().cpu())
            labels = {k: torch.cat(v, dim=0) for k, v in labels.items()}
            outputs = {k: torch.cat(v, dim=0) for k, v in outputs.items()}
        return outputs, labels

    def fit_step(self, batch, *args, **kwargs):
        graph, labels = batch
        pred = self(graph=graph)
        return self._loss(pred, labels)

    def set_loss(self, loss):
        self._loss = loss

    def set_optimizer(self, optimizer):
        self._optimizer = optimizer

    def optimizer(self):
        return self._optimizer

    def fit(self, loader, epoch=10, callback=None):
        if not isinstance(loader, DataLoader):
            print('recieved a not all2graph.DataLoader, function check can not be done')
        else:
            if loader.parser is not self.raw_graph_parser:
                print('loader.parser and module.parser are not the same, which may cause undefined behavior')
        return super().fit(loader=loader, epoch=epoch, callback=callback)
