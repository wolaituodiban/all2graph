import os
from typing import Dict, Union

import numpy as np
import pandas as pd
import torch

from torch.utils.data import DataLoader
from ..data import CSVDataset, GraphDataset
from ..parsers import RawGraphParser, ParserWrapper
from ..graph import Graph, RawGraph
from ..parsers import DataParser
from ..version import __version__
from ..utils import tqdm


def num_parameters(module: torch.nn.Module):
    parameters = {}
    for p in module.parameters():
        parameters[id(p)] = p
    return sum(map(lambda x: np.prod(x.shape), parameters.values()))


class MyModule(torch.nn.Module):
    def __init__(self, raw_graph_parser: RawGraphParser):
        super().__init__()
        self.version = __version__
        self.raw_graph_parser = raw_graph_parser
        self._loss = None
        self._optimizer = None

    def forward(self, inputs) -> Graph:
        if isinstance(inputs, RawGraph):
            inputs = self.raw_graph_parser.parse(inputs)
        return inputs

    @torch.no_grad()
    def predict_dataloader(self, loader: DataLoader, postfix=''):
        if not isinstance(loader.dataset, (CSVDataset, GraphDataset)):
            print('recieved a not all2graph.Dataset, function check can not be done')
        elif isinstance(loader.dataset, CSVDataset) and loader.dataset.raw_graph_parser != self.raw_graph_parser:
            print('raw_graph_parser are not the same, which may cause undefined behavior')
        self.eval()
        labels = None
        outputs = None
        for graph, label in tqdm(loader, postfix=postfix):
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

    def fit(self, loader: DataLoader, **kwargs):
        if not isinstance(loader.dataset, (CSVDataset, GraphDataset)):
            print('recieved a not all2graph.Dataset, function check can not be done')
        elif hasattr(loader, 'dataset')\
                and isinstance(loader.dataset, CSVDataset)\
                and loader.dataset.raw_graph_parser != self.raw_graph_parser:
            print('raw_graph_parser are not the same, which may cause undefined behavior')
        raise NotImplementedError

    def set_filter_key(self, x):
        self.raw_graph_parser.set_filter_key(x)


class Predictor(torch.nn.Module):
    def __init__(self, data_parser: Union[DataParser, Dict[str, DataParser]], module: MyModule):
        super().__init__()
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
        outputs = pd.concat(outputs)
        return outputs

    def set_filter_key(self, x):
        self.module.set_filter_key(x)
