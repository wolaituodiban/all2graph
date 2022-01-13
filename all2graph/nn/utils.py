import copy
import os
import sys
from typing import Dict, Union, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from ..data import Dataset
from ..parsers import RawGraphParser, ParserWrapper
from ..graph import Graph, RawGraph
from ..parsers import DataParser
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


def default_collate(batches):
    fst_item = batches[0]
    if isinstance(fst_item, torch.Tensor):
        if len(fst_item.shape) == 0:
            return torch.stack(batches)
        else:
            return torch.cat(batches, dim=0)
    elif isinstance(fst_item, list):
        return [default_collate(tensors) for tensors in zip(*batches)]
    elif isinstance(fst_item, dict):
        return {key: default_collate([batch[key] for batch in batches]) for key in batches[0]}
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
    def __init__(self, raw_graph_parser: RawGraphParser):
        super().__init__()
        self.version = __version__
        self.raw_graph_parser = raw_graph_parser
        self._loss = None
        self._optimizer = None

    @property
    def device(self):
        raise NotImplementedError

    def forward(self, inputs) -> Tuple[Graph, torch.Tensor]:
        if isinstance(inputs, RawGraph):
            inputs = self.raw_graph_parser.parse(inputs)
        graph = inputs.to(self.device, non_blocking=True)
        target_mask = graph.target_mask(self.raw_graph_parser.target_symbol)
        return graph, target_mask

    def predict_dataloader(self, loader: DataLoader, postfix=None):
        if not hasattr(loader, 'dataset') or not isinstance(loader.dataset, Dataset):
            print('recieved a not all2graph.Dataset, function check can not be done')
        elif isinstance(loader.dataset, Dataset) and loader.dataset.raw_graph_parser != self.raw_graph_parser:
            print('raw_graph_parser are not the same, which may cause undefined behavior')
        return predict_dataloader(self, loader, postfix)

    def fit(self, loader: DataLoader, **kwargs):
        if not isinstance(loader.dataset, Dataset):
            print('recieved a not all2graph.Dataset, function check can not be done')
        elif hasattr(loader, 'dataset')\
                and isinstance(loader.dataset, Dataset)\
                and loader.dataset.raw_graph_parser != self.raw_graph_parser:
            print('raw_graph_parser are not the same, which may cause undefined behavior')
        raise NotImplementedError

    def set_filter_key(self, x):
        self.raw_graph_parser.set_filter_key(x)


class Predictor(torch.nn.Module):
    def __init__(self, data_parser: Union[DataParser, Dict[str, DataParser]], module: Module):
        super().__init__()
        self.version = __version__
        self.data_parser = data_parser
        self.module = module

    def parser_wrapper(self, temp_file, data_parser: DataParser = None):
        return ParserWrapper(data_parser or self.data_parser, self.module.raw_graph_parser, temp_file=temp_file)

    def set_data_parser(self, data_parser: Union[DataParser, Dict[str, DataParser]]):
        self.data_parser = data_parser

    def enable_preprocessing(self):
        if isinstance(self.data_parser, DataParser):
            self.data_parser.enable_preprocessing()
        elif isinstance(self.data_parser, dict):
            for v in self.data_parser.values():
                v.enable_preprocessing()

    def disable_preprocessing(self):
        if isinstance(self.data_parser, DataParser):
            self.data_parser.disable_preprocessing()
        elif isinstance(self.data_parser, dict):
            for v in self.data_parser.values():
                v.disable_preprocessing()

    def forward(self, df: pd.DataFrame) -> Dict[str, torch.Tensor]:
        if self.module.version <= '0.1.5' and __version__ >= '0.1.6':
            print('old version model (<=0.1.5) has a bug in Conv.forward', file=sys.stderr)
            print('from version 0.1.6, that bug has been fixed and output will be influenced', file=sys.stderr)
        parser_wrapper = self.parser_wrapper(temp_file=False)
        df, graphs, *_ = parser_wrapper.parse(df, disable=True)
        return self._predict(graphs)

    def _predict(self, graphs) -> Dict[str, torch.Tensor]:
        outputs = {}
        for name, graph in graphs.items():
            if isinstance(graph, str):
                filename = graph
                graph, _ = Graph.load(graph)
                os.remove(filename)
            pred = self.module(graph)
            for k, v in pred.items():
                outputs['_'.join([k, name])] = v
        return outputs

    @torch.no_grad()
    def predict(
            self, src, chunksize=64, disable=False, processes=0, postfix='predicting', temp_file=False,
            data_parser: DataParser = None, **kwargs) -> pd.DataFrame:
        self.eval()
        if processes == 0:
            temp_file = False
        parser_wrapper = self.parser_wrapper(temp_file=temp_file, data_parser=data_parser)
        outputs = []
        for df, graphs, *_ in parser_wrapper.generate(
                src, chunksize=chunksize, disable=disable, processes=processes, postfix=postfix, **kwargs):
            for k, v in self._predict(graphs).items():
                df[k] = v.cpu().numpy()
            outputs.append(df)
        if len(outputs) == 0:
            return pd.DataFrame()
        outputs = pd.concat(outputs)
        return outputs

    def set_filter_key(self, x):
        self.module.set_filter_key(x)


def _get_activation(act):
    if act == 'relu':
        return torch.nn.ReLU()
    elif act == 'gelu':
        return torch.nn.GELU()
    elif act == 'prelu':
        return torch.nn.PReLU()
    else:
        return copy.deepcopy(act)
