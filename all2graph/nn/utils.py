import copy
from abc import abstractmethod

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from ..data import default_collate
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
        self.register_buffer('_device_tracer', torch.ones(1))
        self.version = __version__

    @property
    def device(self):
        return self._device_tracer.device

    @property
    def num_parameters(self):
        return num_parameters(self)

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
    for graphs, df in parser.generator(src, **kwargs):
        # 将graphs的类型全都转成dict
        if not isinstance(graphs, dict):
            graphs = {'pred': graphs}
        for parser_key, graph in graphs.items():
            # pred的类型全都转成dict
            pred = module(graph)
            if isinstance(pred, list):
                pred = {'output_{}'.format(i): v for i, v in enumerate(pred)}
            elif isinstance(pred, torch.Tensor):
                pred = {'output': pred}
            # 判断value的维度
            for target_key, value in pred.items():
                value = value.view(df.shape[0], -1).squeeze(-1).cpu().numpy()
                if len(value.shape) == 0:
                    continue
                elif len(value.shape) == 1:
                    df['{}_{}'.format(target_key, parser_key)] = value
                else:
                    for j in range(value.shape[1]):
                        df['{}_dim{}_{}'.format(target_key, j, parser_key)] = value[:, j]
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


class Residual(Module):
    def __init__(self, module, pre=None, post=None):
        super().__init__()
        self.module = module
        self.pre = pre
        self.post = post

    def reset_parameters(self):
        if hasattr(self.module, 'reset_parameters'):
            self.module.reset_parameters()
        if hasattr(self.pre, 'reset_parameters'):
            self.pre.reset_parameters()
        if hasattr(self.post, 'reset_parameters'):
            self.post.reset_parameters()

    def forward(self, inputs):
        if self.pre is not None:
            inputs = self.pre(inputs)
        outputs = self.module(inputs)
        if isinstance(outputs, tuple):
            outputs = outputs[0]
        outputs = outputs + inputs
        if self.post is not None:
            outputs = self.post(outputs)
        return outputs
