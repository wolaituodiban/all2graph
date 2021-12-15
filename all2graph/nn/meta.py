import copy
import math
from typing import Dict, List

import dgl
import dgl.function as fn
import numpy as np
import torch
from torch.nn.functional import linear

from .encoder import Encoder
from .utils import num_parameters, MyModule
from ..graph import Graph
from ..meta import MetaNumber
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
    def __init__(
            self, raw_graph_parser: RawGraphParser, encoder: Encoder, num_latent, dropout=0.1,
            norm=True):
        assert raw_graph_parser.num_strings == encoder.value_embedding.num_embeddings
        super().__init__(raw_graph_parser=raw_graph_parser)
        self.param_graph = raw_graph_parser.gen_param_graph(encoder.dynamic_parameter_shapes)
        self.linear = torch.nn.Linear(in_features=encoder.d_model, out_features=num_latent)

        self.name_to_layer = dict()
        self.learners = torch.nn.ModuleList()
        for shape, params in reverse_dict(encoder.dynamic_parameter_shapes).items():
            self.learners.append(BaseMetaLearner(num_latent, shape, dropout=dropout, norm=norm))
            for param in params:
                self.name_to_layer[param] = len(self.learners) - 1

        # 用于卷积元图的主干网络
        self.body = copy.deepcopy(encoder.body)

        # 用于卷积值图的主干网络，学习器需要学习的结构
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
        self.param_graph.to(self.device, non_blocking=True)
        with self.param_graph.graph.local_scope():
            self.param_graph.graph.ndata['emb'] = self.encoder.value_embedding(self.param_graph.value)
            self.param_graph.graph.update_all(fn.copy_u('emb', 'emb'), fn.sum('emb', 'emb'))
            self.param_graph.set_embedding(self.param_graph.graph.ndata['emb'])

    def eval(self):
        self.update_param_emb()
        return super().eval()

    def gen_param(self, param_names: List[str], feat: torch.Tensor = None) -> Dict[str, torch.Tensor]:
        """
        生成底层参数
        Args:
            param_names:
            feat: (*, emb_dim)

        Returns:
            dict of shape(*, *param_shape)
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

    def gen_meta_conv_param(self, meta_graph: dgl.DGLGraph) -> Dict[str, torch.Tensor]:
        """

        Args:
            meta_graph: 元图

        Returns:
            meta_conv需要用到的参数
            (num_blocks, num_nodes, *)
            or (num_blocks, num_edges, *)
        """
        meta_conv_param = self.gen_param(self.body.dynamic_parameter_names)
        for k in self.body.node_dynamic_parameter_names:
            v = meta_conv_param[k]
            meta_conv_param[k] = v.expand(len(self.body), meta_graph.num_nodes(), *v.shape)
        for k in self.body.edge_dynamic_parameter_names:
            v = meta_conv_param[k]
            meta_conv_param[k] = v.expand(len(self.body), meta_graph.num_edges(), *v.shape)
        return meta_conv_param

    def gen_emb_param(self, meta_node_id: torch.Tensor, meta_emb: torch.Tensor) -> Dict[str, torch.Tensor]:
        """

        Args:
            meta_node_id: node对应关系
            meta_emb: graph.meta_value的embedding

        Returns:
            (num_nodes, *)
        """
        emb_param = self.gen_param(self.encoder.node_embedding.dynamic_parameter_names, feat=meta_emb)
        emb_param = {k: v[meta_node_id] for k, v in emb_param.items()}
        return emb_param

    def gen_conv_param(
            self, meta_feats: List[torch.Tensor], meta_values: List[torch.Tensor],
            meta_node_id: torch.Tensor, meta_edge_id: torch.Tensor) -> Dict[str, torch.Tensor]:
        """

        Args:
            meta_feats: 元图卷积之后的list of feats，
                num_blocks of (num_layers, num_meta_nodes, out_dim)
            meta_values: 元图卷积之后的list of values,
                num_blocks of (num_layers, num_meta_edges, out_dim)
            meta_node_id: long, (num_nodes, )
            meta_edge_id: long, (num_edges, )
        Returns:
            (num_blocks, num_nodes, *)
            or (num_blocks, num_edges, *)
        """
        conv_param = {}
        node_param = self.gen_param(
            self.encoder.node_dynamic_parameter_names,
            feat=torch.stack([meta_feat[-1] for meta_feat in meta_feats], dim=0))
        for k, v in node_param.items():
            conv_param[k] = v[:, meta_node_id]
        edge_param = self.gen_param(
            self.encoder.edge_dynamic_parameter_names,
            feat=torch.stack([meta_value[-1] for meta_value in meta_values], dim=0))
        for k, v in edge_param.items():
            conv_param[k] = v[:, meta_edge_id]
        return conv_param

    def gen_output_params(
            self, meta_feats: List[torch.Tensor], meta_node_id: torch.Tensor) -> List[Dict[str, torch.Tensor]]:
        """

        Args:
            meta_feats: 元图卷积之后的list of feats，
                num_blocks of (num_layers, num_meta_nodes, out_dim)
            meta_node_id: long, (num_nodes, )

        Returns:

        """
        # ! 每个block的层数可能不一样，不能直接stack
        output_params = []
        for meta_feat in meta_feats:
            params = self.gen_param(self.encoder.output.dynamic_parameter_names, feat=meta_feat)
            params = {k: v[:, meta_node_id] for k, v in params.items()}
            output_params.append(params)
        return output_params

    def meta_forward(self, graph: Graph):
        """
        输出encoder需要的参数
        Args:
            graph:

        Returns:
        详细信息见各个gen_*_param方法
        """
        # 原因在于Encoder已经不在处理parameter的映射逻辑
        if self.training:
            self.update_param_emb()
        graph = graph.to(self.device, non_blocking=True)
        meta_emb = self.encoder.value_embedding(graph.meta_value)
        # 生成参数meta_conv需要使用的参数
        meta_conv_param = self.gen_meta_conv_param(meta_graph=graph.meta_graph)
        meta_feats, meta_keys, meta_values, meta_attn_weights = self.body(
            graph=graph.meta_graph, in_feat=meta_emb, parameters=meta_conv_param)

        # 生成encoder需要使用的参数
        emb_param = self.gen_emb_param(meta_node_id=graph.meta_node_id, meta_emb=meta_emb)
        conv_param = self.gen_conv_param(
            meta_feats=meta_feats, meta_values=meta_values, meta_node_id=graph.meta_node_id,
            meta_edge_id=graph.meta_edge_id)
        output_params = self.gen_output_params(meta_feats=meta_feats, meta_node_id=graph.meta_node_id)
        return emb_param, conv_param, output_params, {
            'feats': meta_feats, 'keys': meta_keys, 'values': meta_values, 'attn_weights': meta_attn_weights
        }

    def forward(self, graph: Graph, details=False):
        graph = super().forward(graph)
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
        assert raw_graph_parser.num_strings == encoder.value_embedding.num_embeddings, 'parser与encoder不对应'
        super().__init__(raw_graph_parser=raw_graph_parser)
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
        self.reset_parameters(reset_encoder=False)
        self.encoder = encoder

    @property
    def device(self):
        return self.encoder.device

    def reset_parameters(self, reset_encoder=True):
        for param in self.parameters(recurse=False):
            fan = param.shape[-1]
            gain = torch.nn.init.calculate_gain('relu')
            std = gain / math.sqrt(fan)
            bound = math.sqrt(3.0) * std  # Calculate uniform bounds from standard deviation
            torch.nn.init.uniform_(param, -bound, bound)
        if reset_encoder:
            self.encoder.reset_parameters()

    def forward(self, graph: Graph, details=False):
        graph = super().forward(graph)
        graph = graph.to(self.device, non_blocking=True)
        key = graph.key
        emb_param = {
            name: getattr(self, name)[key] for name in self.encoder.node_embedding.dynamic_parameter_names}
        conv_param = {
            name: getattr(self, name)[:, key] for name in self.encoder.body.node_dynamic_parameter_names}
        conv_param.update({
            name: getattr(self, name)[:, graph.edge_key]
            for name in self.encoder.body.edge_dynamic_parameter_names})
        output_params = [
            {
                name: getattr(self, '{}_{}'.format(name, i))[:, key]
                for name in self.encoder.output.dynamic_parameter_names
            }
            for i in range(self.encoder.num_blocks)
        ]
        target_mask = graph.target_mask(self.raw_graph_parser.target_symbol)
        outputs, value_infos = self.encoder(
            graph, emb_param=emb_param, conv_param=conv_param, output_params=output_params,
            target_mask=target_mask, targets=self.raw_graph_parser.targets)
        if details:
            params = {'emb': emb_param, 'conv': conv_param, 'output': output_params}
            return outputs, value_infos, params
        else:
            return outputs

    def extra_repr(self) -> str:
        output = 'num_parameters={}\n'.format(num_parameters(self))
        output += '\n'.join(
            '{}: {}'.format(name, tuple(param.shape)) for name, param in self.named_parameters(recurse=False))
        return output

    @torch.no_grad()
    def load_pretrained(self, other, load_meta_number=False):
        """
        如果预测样本中包含一些字符串，不存在于预训练模型，但是存在于当前模型中，那么预训练模型的结果将无法复现！
        :param other: 预训练
        :param load_meta_number:
            raw_graph_parser会根据内部储存的number分布对number型数据进行归一化处理。如果load_meta_number=True，
            那么会加载预训练模型储存的分布数据，并且复现预训练模型的结果；否则，预测结果会存在一些偏差
        :return:
        """
        # todo 加载size不同的模型
        # todo 进度条显示已经加载的参数量
        if load_meta_number:
            for name, meta_number in other.raw_graph_parser.meta_numbers.items():
                if name in self.raw_graph_parser.meta_numbers:
                    self.raw_graph_parser.meta_numbers[name] = MetaNumber.from_json(meta_number.to_json())

        load_num = 0
        self_num = num_parameters(self)
        for name in other.encoder.output.dynamic_parameter_names:
            for layer_i in range(other.encoder.num_blocks):
                name_i = '{}_{}'.format(name, layer_i)
                if not hasattr(other, name_i) or not hasattr(self, name_i):
                    continue
                for key, key_i in other.raw_graph_parser.key_mapper.items():
                    if key not in self.raw_graph_parser.key_mapper:
                        continue
                    key_j = self.raw_graph_parser.key_mapper[key]
                    temp = getattr(other, name_i)[:, key_i]
                    load_num += np.prod(temp.shape)
                    getattr(self, name_i)[:, key_j] = temp

        for name in other.encoder.node_embedding.node_dynamic_parameter_names:
            if not hasattr(other, name) or not hasattr(self, name):
                continue
            for key, key_i in other.raw_graph_parser.key_mapper.items():
                if key not in self.raw_graph_parser.key_mapper:
                    continue
                key_j = self.raw_graph_parser.key_mapper[key]
                temp = getattr(other, name)[key_i]
                load_num += np.prod(temp.shape)
                getattr(self, name)[key_j] = temp

        for name in other.encoder.body.node_dynamic_parameter_names:
            if not hasattr(other, name) or not hasattr(self, name):
                continue
            for key, key_i in other.raw_graph_parser.key_mapper.items():
                if key not in self.raw_graph_parser.key_mapper:
                    continue
                key_j = self.raw_graph_parser.key_mapper[key]
                temp = getattr(other, name)[:, key_i]
                load_num += np.prod(temp.shape)
                getattr(self, name)[:, key_j] = temp

        for name in other.encoder.body.edge_dynamic_parameter_names:
            if not hasattr(other, name) or not hasattr(self, name):
                continue
            for key, key_i in other.raw_graph_parser.etype_mapper.items():
                if key not in self.raw_graph_parser.etype_mapper:
                    continue
                key_j = self.raw_graph_parser.etype_mapper[key]
                temp = getattr(other, name)[:, key_i]
                load_num += np.prod(temp.shape)
                getattr(self, name)[:, key_j] = temp
        print('{}: {}/{} ({:.1f}%) loaded from pretrained'.format(
            self.__class__.__name__, load_num, self_num, 100*load_num/self_num))
        self.encoder.load_pretrained(other.encoder, self.raw_graph_parser, other.raw_graph_parser)
