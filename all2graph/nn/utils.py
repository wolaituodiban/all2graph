import copy
import os
import sys
from abc import abstractproperty, abstractmethod
from typing import Dict, Union

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from ..data import default_collate
from ..graph import Graph
from ..parsers import DataParser
from ..parsers import ParserWrapper
from ..utils import tqdm
from ..version import __version__


def _get_type(x):
    if isinstance(x, torch.Tensor):
        return 'tensor'
    elif isinstance(x, list):
        return 'list'
    elif isinstance(x, dict):
        return 'dict'
    raise TypeError('only accept "tensor", "list", "dict"')


@torch.no_grad()
def predict_dataloader(module: torch.nn.Module, data_loader: DataLoader, desc=None, max_batch=None):
    """
    仅支持torch.Tensor, list和dict作为输入
    Args:
        module:
        data_loader:
        desc:
        max_batch: 达到最大数量就停止

    Returns:

    """
    module.eval()
    outputs = []
    labels = []
    last_output_type = None
    last_label_type = None
    for i, (graph, label) in enumerate(tqdm(data_loader, desc=desc)):
        if max_batch is not None and i >= max_batch:
            break
        label_type = _get_type(label)
        if last_label_type and label_type != last_label_type:
            raise TypeError('got inconsistent label type: {} and {}'.format(last_label_type, label_type))
        output = module(graph)
        ouptut_type = _get_type(output)
        if last_output_type and ouptut_type != last_output_type:
            raise TypeError('got inconsistent output type: {} and {}'.format(last_output_type, ouptut_type))
        outputs.append(output)
        labels.append(label)
        last_output_type = ouptut_type
        last_label_type = label_type
    return default_collate(outputs), default_collate(labels)


def to_numpy(inputs):
    if isinstance(inputs, list):
        return [to_numpy(x) for x in inputs]
    elif isinstance(inputs, dict):
        return {k: to_numpy(v) for k, v in inputs.items()}
    elif isinstance(inputs, torch.Tensor):
        return inputs.cpu().detach().numpy()
    elif isinstance(inputs, np.ndarray):
        return inputs
    else:
        raise TypeError('only accept "tensor", "list", "dict", "numpy", but got {}'.format(type(inputs)))


def detach(inputs):
    if isinstance(inputs, list):
        return [detach(x) for x in inputs]
    elif isinstance(inputs, dict):
        return {k: detach(v) for k, v in inputs.items()}
    elif isinstance(inputs, torch.Tensor):
        return inputs.cpu().detach()
    else:
        raise TypeError('only accept "tensor", "list", "dict"')


def num_parameters(module: torch.nn.Module):
    parameters = {}
    for p in module.parameters():
        parameters[id(p)] = p
    return sum(map(lambda x: np.prod(x.shape), parameters.values()))


class Module(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.version = __version__

    @abstractproperty
    def device(self):
        raise NotImplementedError

    @property
    def num_parameters(self):
        return num_parameters(self)

    @abstractmethod
    def reset_parameters(self):
        for module in self.children():
            if hasattr(module, 'reset_parameters'):
                module.reset_parameters()

    def predict_dataloader(self, loader: DataLoader, postfix=None):
        return predict_dataloader(self, loader, postfix)

    def extra_repr(self) -> str:
        return 'num_parameters={}'.format(self.num_parameters)


@torch.no_grad()
def predict_csv(parser: ParserWrapper, module: torch.nn.Module, src, **kwargs):
    module.eval()
    dfs = []
    for graphs, df in parser.generator(src, return_df=True, **kwargs):
        if isinstance(graphs, dict):
            for k, graph in graphs.items():
                for kk, pred in module(graph).items():
                    df['{}_{}'.format(kk, k)] = pred.cpu().numpy()
        else:
            for kk, pred in module(graphs).items():
                df['{}_{}'.format(kk, 'pred')] = pred.cpu().numpy()
        dfs.append(df)
    return pd.concat(dfs)


def _get_activation(act):
    if act == 'relu':
        return torch.nn.ReLU()
    elif act == 'gelu':
        return torch.nn.GELU()
    elif act == 'prelu':
        return torch.nn.PReLU()
    else:
        return copy.deepcopy(act)


def _get_norm(norm, *args, **kwargs):
    if norm == 'layer':
        return torch.nn.LayerNorm(*args, **kwargs)
    elif norm == 'batch1d':
        return torch.nn.BatchNorm1d(*args, **kwargs)
    elif norm == 'batch2d':
        return torch.nn.BatchNorm2d(*args, **kwargs)
    elif norm == 'batch3d':
        return torch.nn.BatchNorm3d(*args, **kwargs)
    else:
        return copy.deepcopy(norm)
