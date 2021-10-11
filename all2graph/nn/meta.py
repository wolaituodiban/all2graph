import math
from typing import Dict, Tuple, List

import dgl.function as fn
import torch
from torch.nn.functional import linear

from .utils import num_parameters
from ..parsers import RawGraphParser


class MetaLearnerLayer(torch.nn.Module):
    def __init__(self, num_latent, shape, dropout=0.1, norm=True):
        super().__init__()
        self.dropout = torch.nn.Dropout(dropout)
        self.weight = torch.nn.Parameter(torch.Tensor(*shape, num_latent))
        if norm:
            self.norm = torch.nn.LayerNorm(shape)
        else:
            self.norm = None
        self.reset_parameters()

    @property
    def device(self):
        return self.weight.device

    @property
    def shape(self):
        return tuple(self.weight.shape[:-1])

    @property
    def num_latent(self):
        return self.weight.shape[-1]

    def reset_parameters(self):
        torch.nn.init.kaiming_uniform_(self.weight)
        if self.norm is not None:
            self.norm.reset_parameters()

    def forward(self, emb: torch.Tensor) -> torch.Tensor:
        output = self.dropout(emb)
        weight = self.weight.view(-1, self.num_latent)
        output = linear(output, weight)
        output = output.view(*emb.shape[:-1], *self.shape)
        if self.norm is not None:
            output = self.norm(output)
        return output

    def extra_repr(self) -> str:
        return 'num_latent={}, shape={}, num_parameters={}'.format(
            self.num_latent, self.shape, num_parameters(self))


class MetaLearner(torch.nn.Module):
    def __init__(
            self, raw_graph_parser: RawGraphParser, d_model, num_latent, param_shapes: Dict[Tuple, List[str]],
            dropout=0.1, norm=True):
        super().__init__()
        param_names = []
        for v in param_shapes.values():
            param_names += v
        self.param_graph = raw_graph_parser.gen_param_graph(param_names)
        self.linear = torch.nn.Linear(in_features=d_model, out_features=num_latent)
        self.param_to_layer_id = {}
        self.layers = torch.nn.ModuleList()
        for shape, params in param_shapes.items():
            self.layers.append(MetaLearnerLayer(num_latent, shape, dropout=dropout, norm=norm))
            for param in params:
                self.param_to_layer_id[param] = len(self.layers) - 1

    @property
    def param_shape(self):
        output = {}
        for param, layer_id in self.param_to_layer_id.items():
            shape = self.layers[layer_id].shape
            if shape not in output:
                output[shape] = [param]
            else:
                output[shape].append(param)
        return output

    def reset_parameters(self):
        self.linear.reset_parameters()
        for layer in self.layers:
            layer.reset_parameters()

    def update_embedding(self, emb_module: torch.nn.Module):
        device = list(emb_module.parameters())[0].device
        self.param_graph.graph = self.param_graph.graph.to(device)
        with self.param_graph.graph.local_scope():
            self.param_graph.graph.ndata['emb'] = emb_module(self.param_graph.graph)
            self.param_graph.graph.update_all(fn.copy_u('emb', 'emb'), fn.sum('emb', 'emb'))
            self.param_graph.set_embedding(self.param_graph.graph.ndata['emb'])

    def forward(self, param_names: List[str], feat: torch.Tensor = None) -> Dict[str, torch.Tensor]:
        param_groups = {}
        for param_name in param_names:
            layer_id = self.param_to_layer_id[param_name]
            if layer_id in param_groups:
                param_groups[layer_id].append(param_name)
            else:
                param_groups[layer_id] = [param_name]
        output = {}
        for layer_id, param_names in param_groups.items():
            param_emb = self.param_graph.get_embedding(param_names)
            if feat is not None:
                param_emb = param_emb.view(param_emb.shape[0], 1, param_emb.shape[1]) + feat.view(1, *feat.shape)
            param_emb = self.linear(param_emb)
            params = self.layers[layer_id](param_emb)
            for param_name, param in zip(param_names, params):
                output[param_name] = param
        return output

    def extra_repr(self) -> str:
        return '\n'.join('{}: {}'.format(shape, params) for shape, params in self.param_shape.items())


class MockMetaLearner(torch.nn.Module):
    def __init__(
            self, num_etypes: int, edge_param_shapes: Dict[Tuple, List[str]], num_ntypes: int,
            node_param_shapes: Dict[Tuple, List[str]]):
        super().__init__()
        for shape, params in edge_param_shapes.items():
            for param in params:
                tensor = torch.Tensor(num_etypes, *shape)
                self.register_parameter(param, torch.nn.Parameter(tensor))
        for shape, params in node_param_shapes.items():
            for param in params:
                tensor = torch.Tensor(num_ntypes, *shape)
                self.register_parameter(param, torch.nn.Parameter(tensor))
        self.reset_parameters()

    def reset_parameters(self):
        for param in self.parameters():
            fan = param.shape[-1]
            gain = torch.nn.init.calculate_gain('relu')
            std = gain / math.sqrt(fan)
            bound = math.sqrt(3.0) * std  # Calculate uniform bounds from standard deviation
            torch.nn.init.uniform_(param, -bound, bound)

    def update_embedding(self, emb_module: torch.nn.Module):
        pass

    def forward(self, param_names: List[str], feat: torch.Tensor = None) -> Dict[str, torch.Tensor]:
        if feat is None:
            return {}
        else:
            return {name: getattr(self, name)[feat] for name in param_names}

    def extra_repr(self) -> str:
        output = 'num_parameters={}\n'.format(num_parameters(self))
        output += '\n'.join('{}: {}'.format(name, tuple(param.shape)) for name, param in self.named_parameters())
        return output
