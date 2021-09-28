import copy
import math
from typing import List, Dict, Union

import dgl
import dgl.function as fn
import torch
from torch.utils.data import DataLoader
from toad.nn import Module

from .conv import HeteroAttnConv
from .embedding import ValueEmbedding
from .meta import MetaLearner
from .target import Target
from .utils import num_parameters
from ..globals import FEATURE, SEP, VALUE, META_NODE_ID, META_EDGE_ID, TARGET, KEY, TYPE
from ..graph import Graph
from ..graph.parser import GraphParser
from ..utils import progress_wrapper


def _gen_conv_blocks(
        num_layers, d_model, dropout, node_activation, key_activation, value_activation, residual, edge_bias,
        node_bias, share_conv, norm, node_norm, key_norm, value_norm
):
    blocks = torch.nn.ModuleList()
    base_conv = HeteroAttnConv(
        d_model, dropout=dropout, node_activation=node_activation, key_activation=key_activation,
        value_activation=value_activation, residual=residual, edge_bias=edge_bias, node_bias=node_bias, norm=norm,
        node_norm=node_norm, key_norm=key_norm, value_norm=value_norm
    )
    for n1 in num_layers:
        if share_conv:
            conv = torch.nn.ModuleList([base_conv] * n1)
        else:
            conv = torch.nn.ModuleList([copy.deepcopy(base_conv) for _ in range(n1)])
        blocks.append(conv)
    return blocks


class GFMEncoder(Module):
    """graph factorization machine"""
    def __init__(self, graph_parser: GraphParser, d_model, nhead, num_layers: List[int],
                 dropout=0.1, node_activation='relu', key_activation=None, value_activation=None, conv_residual=False,
                 block_residual=True, share_conv=False, norm=True, edge_bias=True, node_bias=True, node_norm=False,
                 key_norm=False, value_norm=False, **kwargs):
        super().__init__()
        self.graph_parser = graph_parser
        self.embedding = ValueEmbedding(num_embeddings=graph_parser.num_strings, embedding_dim=d_model, **kwargs)
        self.blocks = _gen_conv_blocks(
            num_layers=num_layers, d_model=d_model, dropout=dropout, node_activation=node_activation,
            key_activation=key_activation, value_activation=value_activation, residual=conv_residual,
            edge_bias=edge_bias, node_bias=node_bias, share_conv=share_conv, norm=norm, node_norm=node_norm,
            key_norm=key_norm, value_norm=value_norm)
        self.block_residual = block_residual
        self.target = Target(targets=torch.tensor([self.graph_parser.encode(TARGET)]))
        self.loss = None

        self._init_model(nhead)
        self._optimizer_config = {}

    @property
    def d_model(self):
        return self.embedding.embedding.weight.shape[1]

    @property
    def num_layers(self):
        return len(self.blocks)

    @property
    def num_keys(self):
        return self.graph_parser.num_keys

    @property
    def num_etypes(self):
        return self.graph_parser.num_etypes

    @property
    def device(self):
        return self.embedding.device

    def _init_model(self, nhead):
        # 参数
        d_per_head = self.d_model // nhead
        for param_name in HeteroAttnConv.NODE_PARAMS_2D:
            param = torch.nn.Parameter(torch.Tensor(self.num_layers, self.num_keys, nhead, d_per_head, self.d_model))
            setattr(self, param_name, param)
        for param_name in HeteroAttnConv.NODE_PARAMS_1D + [Target.TARGET_WEIGHT]:
            param = torch.nn.Parameter(torch.Tensor(self.num_layers, self.num_keys, nhead, d_per_head))
            setattr(self, param_name, param)

        for param_name in HeteroAttnConv.EDGE_PARAMS_2D:
            param = torch.nn.Parameter(torch.Tensor(self.num_layers, self.num_etypes, nhead, d_per_head, self.d_model))
            setattr(self, param_name, param)
        for param_name in HeteroAttnConv.EDGE_PARAMS_1D:
            param = torch.nn.Parameter(torch.Tensor(self.num_layers, self.num_etypes, nhead, d_per_head))
            setattr(self, param_name, param)
        self.reset_parameters()

    def set_loss(self, loss):
        self.loss = loss

    def set_optimizer_config(self, config=None):
        self._optimizer_config = dict(config or {})

    def optimizer(self):
        return torch.optim.AdamW(self.parameters(), **self._optimizer_config)

    def reset_parameters(self):
        fan = self.embedding.embedding.weight.shape[1]
        gain = torch.nn.init.calculate_gain('leaky_relu')
        std = gain / math.sqrt(fan)
        bound = math.sqrt(3.0) * std  # Calculate uniform bounds from standard deviation
        for param in HeteroAttnConv.NODE_PARAMS_2D + HeteroAttnConv.EDGE_PARAMS_2D:
            torch.nn.init.uniform_(getattr(self, param), -bound, bound)
        bound = 1 / math.sqrt(fan)
        for param in HeteroAttnConv.NODE_PARAMS_1D + HeteroAttnConv.EDGE_PARAMS_1D + [Target.TARGET_WEIGHT]:
            torch.nn.init.uniform_(getattr(self, param), -bound, bound)

        for module in self.children():
            if hasattr(module, 'reset_parameters'):
                module.reset_parameters()

    def _asign_param_to_graph(self, graph, i_layer):
        for param in HeteroAttnConv.NODE_PARAMS_1D + HeteroAttnConv.NODE_PARAMS_2D + [Target.TARGET_WEIGHT]:
            graph.ndata[param] = getattr(self, param)[i_layer][graph.ndata[KEY]]
        for param in HeteroAttnConv.EDGE_PARAMS_1D + HeteroAttnConv.EDGE_PARAMS_2D:
            graph.edata[param] = getattr(self, param)[i_layer][graph.edata[TYPE]]

    def forward(self, graph: Union[dgl.DGLGraph, Graph], details=False):
        if isinstance(graph, Graph):
            self.graph_parser.set_meta_mode(False)
            graph = self.graph_parser.graph_to_dgl(graph)
        graph = graph.to(self.device)

        output = []
        feats = [[] for _ in range(len(self.blocks)+1)]
        keys = [[] for _ in range(len(self.blocks))]
        values = [[] for _ in range(len(self.blocks))]
        attn_weights = [[] for _ in range(len(self.blocks))]
        with graph.local_scope():
            feat = self.embedding(graph)
            feats[0].append(feat)
            for i, block in enumerate(self.blocks):
                self._asign_param_to_graph(graph, i)
                for conv in block:
                    feat, key, value, attn_weight = conv(graph, feat)
                    output.append(self.target(graph, feat))  # (num_components * num_targets, )
                    feats[i+1].append(feat)  # 多存了一层初始的embedding
                    keys[i].append(key)
                    values[i].append(value)
                    attn_weights[i].append(attn_weight)
                if self.block_residual:
                    feat = feat + feats[i][-1]
        output = torch.stack(output, -1)
        output = torch.mean(output, -1)
        output = output.view(-1, len(self.graph_parser.targets))
        output = {target: output[:, i] for i, target in enumerate(self.graph_parser.targets)}
        if details:
            return output, feats, keys, values, attn_weights
        else:
            return output

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
        return self.loss(pred, labels)

    def extra_repr(self) -> str:
        return 'block_residual={}, num_parameters={}\ngraph_parser={}'.format(
            self.block_residual, num_parameters(self), self.graph_parser)


class UGFMEncoder(GFMEncoder):
    """universal graph factorization machine"""
    def __init__(self, graph_parser: GraphParser, d_model, num_latent, nhead, num_layers: List[int],
                 num_meta_layers: List[int], dropout=0.1, node_activation='relu', key_activation=None,
                 value_activation=None, conv_residual=False, block_residual=True, share_conv=False, norm=True,
                 edge_bias=True, node_bias=True, node_norm=False, key_norm=False, value_norm=False, **kwargs):
        assert len(num_layers) == len(num_meta_layers)
        super(UGFMEncoder, self).__init__(
            graph_parser=graph_parser, d_model=d_model, nhead=nhead, num_layers=num_layers, dropout=dropout,
            node_activation=node_activation, key_activation=key_activation, value_activation=value_activation,
            conv_residual=conv_residual, block_residual=block_residual, share_conv=share_conv, norm=norm,
            edge_bias=edge_bias, node_bias=node_bias, node_norm=node_norm, key_norm=key_norm, value_norm=value_norm,
            **kwargs)
        self.meta_learner = MetaLearner(d_model, num_latent, nhead, d_model // nhead, dropout=dropout)
        self.meta_blocks = _gen_conv_blocks(
            num_layers=num_meta_layers, d_model=d_model, dropout=dropout,
            node_activation=node_activation, key_activation=key_activation, value_activation=value_activation,
            residual=conv_residual, edge_bias=edge_bias, node_bias=node_bias, share_conv=share_conv, norm=norm,
            node_norm=node_norm, key_norm=key_norm, value_norm=value_norm)

        self._meta_feats = None
        self._meta_values = None

    def reset_parameters(self):
        for module in self.children():
            if hasattr(module, 'reset_parameters'):
                module.reset_parameters()

    def _init_model(self, nhead):
        """
        初始化参数的名字组成的图
        """
        self._node_param_1d_idx: Dict[str, int] = {}
        self._node_param_2d_idx: Dict[str, int] = {}
        self._edge_param_1d_idx: Dict[str, int] = {}
        self._edge_param_2d_idx: Dict[str, int] = {}
        src, dst = [], []
        value = []

        def init_func(param_list: List[str], index_dict: Dict[str, int]):
            for name in param_list:
                index_dict[name] = len(value)
                value.append(name)
                for token in name.split(SEP):
                    src.append(len(value))
                    dst.append(index_dict[name])
                    value.append(token)

        init_func([Target.TARGET_WEIGHT], self._node_param_1d_idx)
        init_func(HeteroAttnConv.NODE_PARAMS_1D, self._node_param_1d_idx)
        init_func(HeteroAttnConv.NODE_PARAMS_2D, self._node_param_2d_idx)
        init_func(HeteroAttnConv.EDGE_PARAMS_1D, self._edge_param_1d_idx)
        init_func(HeteroAttnConv.EDGE_PARAMS_2D, self._edge_param_2d_idx)

        self._param_key_graph = dgl.graph((src, dst), num_nodes=len(value))
        self._param_key_graph.ndata[VALUE] = torch.tensor(
            list(map(self.graph_parser.encode, value)), dtype=torch.long)

    def _init_base_param(self):
        """根据参数名字图，生成基本参数"""
        self._param_key_graph = self._param_key_graph.to(self.device)
        with self._param_key_graph.local_scope():
            self._param_key_graph.ndata[FEATURE] = self.embedding(self._param_key_graph)
            self._param_key_graph.update_all(fn.copy_u(FEATURE, FEATURE), fn.sum(FEATURE, FEATURE))
            param_2d, param_1d = self.meta_learner(self._param_key_graph.ndata[FEATURE])

            def init_func(param: torch.Tensor, *idx_dicts: Dict[str, int]):
                for idx_dict in idx_dicts:
                    for key, idx in idx_dict.items():
                        self.register_buffer(key, param[idx], persistent=False)

            init_func(param_2d, self._node_param_2d_idx, self._edge_param_2d_idx)
            init_func(param_1d, self._node_param_1d_idx, self._edge_param_1d_idx)

        self._meta_feats = []
        self._meta_values = []

    def eval(self):
        self._init_base_param()
        return super().eval()

    def _asign_param_to_graph(self, graph, i_layer=None):
        for key, param in self.named_buffers(recurse=False):
            if key in self._node_param_2d_idx or key in self._node_param_1d_idx:
                graph.ndata[key] = param.repeat(graph.num_nodes(), *[1] * len(param.shape))
            elif key in self._edge_param_2d_idx or key in self._edge_param_1d_idx:
                graph.edata[key] = param.repeat(graph.num_edges(), *[1] * len(param.shape))

        # 将元图的embedding生成参数，并加到基本参数上
        if i_layer is not None:
            node_param_2d, node_param_1d = self.meta_learner(self._meta_feats[i_layer+1][-1])  # 多存了一层初始的embedding
            for key in self._node_param_2d_idx:
                graph.ndata[key] += node_param_2d[graph.ndata[META_NODE_ID]]
            for key in self._node_param_1d_idx:
                graph.ndata[key] += node_param_1d[graph.ndata[META_NODE_ID]]

            edge_param_2d, edge_param_1d = self.meta_learner(self._meta_values[i_layer][-1])
            for key in self._edge_param_2d_idx:
                graph.edata[key] += edge_param_2d[graph.edata[META_EDGE_ID]]
            for key in self._edge_param_1d_idx:
                graph.edata[key] += edge_param_1d[graph.edata[META_EDGE_ID]]

    def forward(self, graph: Union[dgl.DGLGraph, Graph], meta_graph: dgl.DGLGraph = None, details=False):
        if self.training:
            self._init_base_param()

        if meta_graph is None:
            self.graph_parser.set_meta_mode(True)
            meta_graph, graph = self.graph_parser.graph_to_dgl(graph)
        meta_graph = meta_graph.to(self.device)

        # 生成元图的embedding
        self._meta_feats = [[] for _ in range(len(self.blocks)+1)]  # 多存了一层初始的embedding
        meta_keys = [[] for _ in range(len(self.blocks))]
        self._meta_values = [[] for _ in range(len(self.blocks))]
        meta_attn_weights = [[] for _ in range(len(self.blocks))]
        with meta_graph.local_scope():
            meta_feat = self.embedding(meta_graph)
            self._meta_feats[0].append(meta_feat)
            self._asign_param_to_graph(meta_graph)
            for i, block in enumerate(self.meta_blocks):
                for meta_conv in block:
                    meta_feat, meta_key, meta_value, meta_attn_weight = meta_conv(meta_graph, meta_feat)
                    self._meta_feats[i+1].append(meta_feat)  # 多存了一层初始的embedding
                    meta_keys[i].append(meta_key)
                    self._meta_values[i].append(meta_value)
                    meta_attn_weights[i].append(meta_attn_weight)
                if self.block_residual:
                    meta_feat = meta_feat + self._meta_feats[i][-1]
        output, feats, keys, values, attn_weights = super().forward(graph, details=True)
        if details:
            return output, self._meta_feats, meta_keys, self._meta_values, meta_attn_weights, \
                   feats, keys, values, attn_weights
        else:
            return output

    def predict_dataloader(self, dataloader: DataLoader, postfix=''):
        with torch.no_grad():
            self.eval()
            labels = None
            outputs = None
            for (meta_graph, graph), label in progress_wrapper(dataloader, postfix=postfix):
                output = self(meta_graph=meta_graph, graph=graph)
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
        (meta_graph, graph), labels = batch
        pred = self(graph=graph, meta_graph=meta_graph)
        return self.loss(pred, labels)

    def extra_repr(self) -> str:
        return '{}\nparam_key_graph={}'.format(super().extra_repr(), self._param_key_graph)
