import copy
import math
from typing import Dict, List, Union

import dgl.function as fn
import torch
from torch.nn.functional import linear

from .encoder import Encoder
from .utils import num_parameters, MyModule
from ..graph import RawGraph, Graph
from ..parsers import RawGraphParser


class BaseMetaLearner(torch.nn.Module):
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


def reverse_dict(d: dict):
    output = {}
    for k, v in d.items():
        if v in output:
            output[v].append(k)
        else:
            output[v] = [k]
    return output


class EncoderMetaLearner(MyModule):
    def __init__(self, raw_graph_parser: RawGraphParser, encoder: Encoder, num_latent, dropout=0.1, norm=True):
        super().__init__()
        self.raw_graph_parser = raw_graph_parser
        self.param_graph = raw_graph_parser.gen_param_graph(encoder.dynamic_parameter_shapes)
        self.linear = torch.nn.Linear(in_features=encoder.d_model, out_features=num_latent)

        self.name_to_layer = dict()
        self.learners = torch.nn.ModuleList()
        for shape, params in reverse_dict(encoder.dynamic_parameter_shapes).items():
            self.learners.append(BaseMetaLearner(num_latent, shape, dropout=dropout, norm=norm))
            for param in params:
                self.name_to_layer[param] = len(self.learners) - 1
        self.body = copy.deepcopy(encoder.body)
        self.encoder = encoder

    @property
    def num_blocks(self):
        return len(self.body)

    @property
    def device(self):
        return self.encoder.device

    @property
    def shape_to_name(self):
        return reverse_dict(self.encoder.dynamic_parameter_shapes)

    def reset_parameters(self):
        self.encoder.reset_parameters()
        self.linear.reset_parameters()
        for layer in self.learners:
            layer.reset_parameters()

    def update_param_emb(self):
        self.param_graph.to(self.device)
        with self.param_graph.graph.local_scope():
            self.param_graph.graph.ndata['emb'] = self.encoder.value_embedding(self.param_graph.value)
            self.param_graph.graph.update_all(fn.copy_u('emb', 'emb'), fn.sum('emb', 'emb'))
            self.param_graph.set_embedding(self.param_graph.graph.ndata['emb'])

    def eval(self):
        self.update_param_emb()
        return super().eval()

    def gen_param(self, param_names: List[str], feat: torch.Tensor = None) -> Dict[str, torch.Tensor]:
        """

        :param param_names:
        :param feat: (*, emb_dim)
        :return: dict of shape(*, *param_shape)
        """
        layer_to_name = reverse_dict({name: self.name_to_layer[name] for name in param_names})
        output = {}
        if feat is not None:
            feat = feat.view(1, *feat.shape)  # (*, 1, emb_dim)
        for layer_id, param_names in layer_to_name.items():
            param_emb = self.param_graph.get_embedding(param_names)  # (num_params, emb_dim)
            if feat is not None:
                param_emb = param_emb.view(param_emb.shape[0], *[1]*(len(feat.shape)-2), param_emb.shape[1])
                param_emb = param_emb + feat  # (*, num_params, emb_dim)
            param_emb = self.linear(param_emb)  # (*, num_params, num_latent)
            params = self.learners[layer_id](param_emb)  # (*, num_params, *param_shape)
            for param_name, param in zip(param_names, params):
                output[param_name] = param
        return output

    def meta_forward(self, graph: Graph):
        if self.training:
            self.update_param_emb()

        meta_graph = graph.meta_graph.to(self.device)
        meta_emb = self.encoder.value_embedding(graph.meta_value)
        meta_conv_param = self.gen_param(self.encoder.body.dynamic_parameter_names)
        meta_conv_param = {k: [v] * self.num_blocks for k, v in meta_conv_param.items()}
        meta_feats, meta_keys, meta_values, meta_attn_weights = self.body(
            graph=meta_graph, in_feat=meta_emb, parameters=meta_conv_param)
        emb_param = self.gen_param(self.encoder.node_embedding.dynamic_parameter_names, feat=meta_emb)
        conv_param = self.gen_param(
            self.encoder.node_dynamic_parameter_names,
            feat=torch.stack([meta_feat[-1] for meta_feat in meta_feats], dim=0))
        conv_edge_param = self.gen_param(
            self.encoder.edge_dynamic_parameter_names,
            feat=torch.stack([meta_value[-1] for meta_value in meta_values], dim=0))
        conv_param.update(conv_edge_param)
        output_params = [
            self.gen_param(self.encoder.output.dynamic_parameter_names, feat=meta_feat) for meta_feat in meta_feats]
        return emb_param, conv_param, output_params, {
            'feats': meta_feats, 'keys': meta_keys, 'values': meta_values, 'attn_weights': meta_attn_weights
        }

    def forward(self, graph: Union[RawGraph, Graph], details=False):
        if isinstance(graph, RawGraph):
            graph = self.raw_graph_parser.parse(graph)
        emb_param, conv_param, output_params, meta_infos = self.meta_forward(graph)
        target_mask = graph.target_mask(self.raw_graph_parser.target_symbol)
        outputs, value_infos = self.encoder(
            graph, emb_param=emb_param, conv_param=conv_param, output_params=output_params, target_mask=target_mask,
            targets=self.raw_graph_parser.targets
        )
        if details:
            return outputs, meta_infos, emb_param, conv_param, output_params, value_infos
        else:
            return outputs

    def extra_repr(self) -> str:
        output = 'num_parameters={}\n'.format(num_parameters(self))
        output += '\n'.join('{}: {}'.format(shape, params) for shape, params in self.shape_to_name.items())
        return output


class EncoderMetaLearnerMocker(MyModule):
    def __init__(self, raw_graph_parser: RawGraphParser, encoder: Encoder):
        super().__init__()
        self.raw_graph_parser = raw_graph_parser
        for name, shape in encoder.dynamic_parameter_shapes.items():
            if name in encoder.output.dynamic_parameter_names:
                for i, num_layers in enumerate(encoder.num_layers):
                    tensor = torch.Tensor(num_layers, raw_graph_parser.num_keys, *shape)
                    self.register_parameter('{}_{}'.format(name, i), torch.nn.Parameter(tensor))

            elif name in encoder.node_embedding.node_dynamic_parameter_names:
                tensor = torch.Tensor(self.raw_graph_parser.num_keys, *shape)
                self.register_parameter(name, torch.nn.Parameter(tensor))

            elif name in encoder.body.node_dynamic_parameter_names:
                tensor = torch.Tensor(encoder.num_blocks, self.raw_graph_parser.num_keys, *shape)
                self.register_parameter(name, torch.nn.Parameter(tensor))

            elif name in encoder.body.edge_dynamic_parameter_names:
                tensor = torch.Tensor(encoder.num_blocks, raw_graph_parser.num_etypes, *shape)
                self.register_parameter(name, torch.nn.Parameter(tensor))

            else:
                raise KeyError('unknown parameter name ({}), check source code!'.format(name))
        self.encoder = encoder

    def reset_parameters(self):
        for param in self.parameters():
            fan = param.shape[-1]
            gain = torch.nn.init.calculate_gain('relu')
            std = gain / math.sqrt(fan)
            bound = math.sqrt(3.0) * std  # Calculate uniform bounds from standard deviation
            torch.nn.init.uniform_(param, -bound, bound)

    def forward(self, graph: Union[RawGraph, Graph], details=False):
        if isinstance(graph, RawGraph):
            graph = self.raw_graph_parser.parse(graph)

        emb_param = {name: getattr(self, name) for name in self.encoder.node_embedding.dynamic_parameter_names}
        conv_param = {name: getattr(self, name) for name in self.encoder.body.dynamic_parameter_names}
        output_params = [
            {name: getattr(self, '{}_{}'.format(name, i)) for name in self.encoder.output.dynamic_parameter_names}
            for i in range(self.encoder.num_blocks)
        ]
        target_mask = graph.target_mask(self.raw_graph_parser.target_symbol)
        outputs, value_infos = self.encoder(
            graph, emb_param=emb_param, conv_param=conv_param, output_params=output_params,
            target_mask=target_mask, targets=self.raw_graph_parser.targets)
        if details:
            return outputs, value_infos
        else:
            return outputs

    def extra_repr(self) -> str:
        output = 'num_parameters={}\n'.format(num_parameters(self))
        output += '\n'.join(
            '{}: {}'.format(name, tuple(param.shape)) for name, param in self.named_parameters(recurse=False))
        return output
