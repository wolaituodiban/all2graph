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
from ..globals import FEATURE, VALUE, META_NODE_ID, META_EDGE_ID, TARGET, KEY, TYPE, BIAS, META
from ..graph import Graph
from ..graph.parser import GraphParser
from ..utils import progress_wrapper
from ..version import __version__


def _gen_conv_blocks(
        num_layers, d_model, dropout, key_norm, key_activation, value_norm, value_activation,
        node_norm, node_activation, residual, norm, share_conv
):
    blocks = torch.nn.ModuleList()
    base_conv = HeteroAttnConv(
        d_model, dropout=dropout, key_norm=key_norm, key_activation=key_activation, value_norm=value_norm,
        value_activation=value_activation, node_norm=node_norm, node_activation=node_activation, residual=residual,
        norm=norm)
    for n1 in num_layers:
        if share_conv:
            conv = torch.nn.ModuleList([base_conv] * n1)
        else:
            conv = torch.nn.ModuleList([copy.deepcopy(base_conv) for _ in range(n1)])
        blocks.append(conv)
    return blocks


class GFMEncoder(Module):
    """graph factorization machine"""
    def __init__(self, graph_parser: GraphParser, d_model, nhead, num_layers: List[int], dropout=0.1, edge_bias=False,
                 key_norm=False, key_activation=None, value_norm=False, value_activation=None, node_bias=False,
                 node_norm=False, node_activation='relu', conv_residual=False, norm=True, share_conv=False,
                 block_residual=True, in_block_output=False, **kwargs):
        """

        :param graph_parser:
        :param d_model:
        :param nhead:
        :param num_layers:
        :param dropout:
        :param edge_bias:
        :param key_norm:
        :param key_activation:
        :param value_norm:
        :param value_activation:
        :param node_bias:
        :param node_norm:
        :param node_activation:
        :param conv_residual: block之内残差连接
        :param norm:
        :param share_conv: 共享卷积层
        :param block_residual: block之间残差连接
        :param in_block_output: block之内输出
        :param kwargs:
        """
        super().__init__()
        self.version = __version__
        self.graph_parser = graph_parser
        self.embedding = ValueEmbedding(num_embeddings=graph_parser.num_strings, embedding_dim=d_model, **kwargs)
        self.blocks = _gen_conv_blocks(
            num_layers=num_layers, d_model=d_model, dropout=dropout, node_activation=node_activation,
            key_activation=key_activation, value_activation=value_activation, residual=conv_residual,
            share_conv=share_conv, norm=norm, node_norm=node_norm, key_norm=key_norm, value_norm=value_norm)
        self.block_residual = block_residual
        self.in_block_output = in_block_output
        self.output = Target(targets=torch.tensor([self.graph_parser.encode(TARGET)]))
        self.loss = None

        self._init_model(nhead, edge_bias=edge_bias, node_bias=node_bias)
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

    @property
    def use_edge_bias(self):
        for name in self.named_parameters(recurse=False):
            if name in HeteroAttnConv.EDGE_PARAMS_1D and BIAS in name:
                return True
        return False

    @property
    def use_node_bias(self):
        for name in self.named_parameters(recurse=False):
            if name in HeteroAttnConv.NODE_PARAMS_1D and BIAS in name:
                return True
        return False

    def set_loss(self, loss):
        self.loss = loss

    def set_optimizer_config(self, config=None):
        self._optimizer_config = dict(config or {})

    def optimizer(self):
        return torch.optim.AdamW(self.parameters(), **self._optimizer_config)

    def _init_model(self, nhead, edge_bias, node_bias):
        # 参数
        assert self.d_model % nhead == 0
        d_per_head = self.d_model // nhead
        for param_name in HeteroAttnConv.EDGE_PARAMS_2D:
            param = torch.nn.Parameter(torch.Tensor(self.num_layers, self.num_etypes, nhead, d_per_head, self.d_model))
            setattr(self, param_name, param)
        for param_name in HeteroAttnConv.EDGE_PARAMS_1D:
            if not edge_bias and BIAS in param_name:
                continue
            param = torch.nn.Parameter(torch.Tensor(self.num_layers, self.num_etypes, nhead, d_per_head))
            setattr(self, param_name, param)

        for param_name in HeteroAttnConv.NODE_PARAMS_2D:
            param = torch.nn.Parameter(torch.Tensor(self.num_layers, self.num_keys, nhead, d_per_head, self.d_model))
            setattr(self, param_name, param)
        for param_name in HeteroAttnConv.NODE_PARAMS_1D + [Target.TARGET_WEIGHT]:
            if not node_bias and BIAS in param_name:
                continue
            param = torch.nn.Parameter(torch.Tensor(self.num_layers, self.num_keys, nhead, d_per_head))
            setattr(self, param_name, param)

        self.reset_parameters()

    def reset_parameters(self):
        fan = self.embedding.embedding.weight.shape[1]
        gain = torch.nn.init.calculate_gain('leaky_relu')
        std = gain / math.sqrt(fan)
        bound = math.sqrt(3.0) * std  # Calculate uniform bounds from standard deviation
        for param in HeteroAttnConv.NODE_PARAMS_2D + HeteroAttnConv.EDGE_PARAMS_2D:
            torch.nn.init.uniform_(getattr(self, param), -bound, bound)
        bound = 1 / math.sqrt(fan)
        for param in HeteroAttnConv.NODE_PARAMS_1D + HeteroAttnConv.EDGE_PARAMS_1D + [Target.TARGET_WEIGHT]:
            if hasattr(self, param):
                torch.nn.init.uniform_(getattr(self, param), -bound, bound)

        for module in self.children():
            if hasattr(module, 'reset_parameters'):
                module.reset_parameters()

    def _asign_param_to_graph(self, graph, i_layer):
        for param_name in HeteroAttnConv.NODE_PARAMS_1D + HeteroAttnConv.NODE_PARAMS_2D + [Target.TARGET_WEIGHT]:
            if hasattr(self, param_name):
                graph.ndata[param_name] = getattr(self, param_name)[i_layer, graph.ndata[KEY]]
        for param_name in HeteroAttnConv.EDGE_PARAMS_1D + HeteroAttnConv.EDGE_PARAMS_2D:
            if hasattr(self, param_name):
                graph.edata[param_name] = getattr(self, param_name)[i_layer, graph.edata[TYPE]]

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
                    feats[i+1].append(feat)  # 多存了一层初始的embedding
                    keys[i].append(key)
                    values[i].append(value)
                    attn_weights[i].append(attn_weight)
                    if self.in_block_output:
                        output.append(self.output(graph, feat))  # (num_components * num_targets, )
                if self.block_residual:
                    feat = feat + feats[i][-1]
                output.append(self.output(graph, feat))  # (num_components * num_targets, )
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
        s = 'num_parameters={}, edge_bias={}, node_bias={}, block_residual={}, in_block_output={}\ngraph_parser={}'
        return s.format(num_parameters(self), self.use_edge_bias, self.use_node_bias, self.block_residual, self.in_block_output,
                        self.graph_parser)


class UGFMEncoder(GFMEncoder):
    """universal graph factorization machine"""
    def __init__(self, graph_parser: GraphParser, d_model, num_latent, nhead, num_layers: List[int],
                 num_meta_layers: List[int], dropout=0.1, edge_bias=False, key_norm=False, key_activation=None,
                 value_norm=False, value_activation=None, node_bias=False, node_norm=False, node_activation='relu',
                 conv_residual=False, norm=True, share_conv=False, block_residual=True, in_block_output=False, **kwargs):
        assert len(num_layers) == len(num_meta_layers)
        super(UGFMEncoder, self).__init__(
            graph_parser=graph_parser, d_model=d_model, nhead=nhead, num_layers=num_layers, dropout=dropout,
            node_activation=node_activation, key_activation=key_activation, value_activation=value_activation,
            conv_residual=conv_residual, block_residual=block_residual, share_conv=share_conv, norm=norm,
            edge_bias=edge_bias, node_bias=node_bias, node_norm=node_norm, key_norm=key_norm, value_norm=value_norm,
            in_block_output=in_block_output, **kwargs)
        self.meta_learner = MetaLearner(d_model, num_latent, nhead, d_model // nhead, dropout=dropout)
        self.meta_blocks = _gen_conv_blocks(
            num_layers=num_meta_layers, d_model=d_model, dropout=dropout,
            node_activation=node_activation, key_activation=key_activation, value_activation=value_activation,
            residual=conv_residual, share_conv=share_conv, norm=norm,
            node_norm=node_norm, key_norm=key_norm, value_norm=value_norm)

    @property
    def use_edge_bias(self):
        for name in self._edge_param_1d_idx:
            if BIAS in name:
                return True
        return False

    @property
    def use_node_bias(self):
        for name in self._node_param_1d_idx:
            if BIAS in name:
                return True
        return False

    def reset_parameters(self):
        for module in self.children():
            if hasattr(module, 'reset_parameters'):
                module.reset_parameters()

    def _init_model(self, nhead, edge_bias, node_bias):
        """
        初始化参数的名字组成的图
        """
        all_params = [Target.TARGET_WEIGHT] + HeteroAttnConv.NODE_PARAMS_2D + HeteroAttnConv.EDGE_PARAMS_2D
        all_params += [x for x in HeteroAttnConv.EDGE_PARAMS_1D if edge_bias or BIAS not in x]
        all_params += [x for x in HeteroAttnConv.NODE_PARAMS_1D if node_bias or BIAS not in x]

        self._param_key_graph = Graph()
        for param in all_params:
            if all_params not in self._param_key_graph.value:
                param_id = self._param_key_graph.insert_node(0, key=KEY, value=param, self_loop=False, type=KEY)
                if param in self.graph_parser.string_mapper:
                    continue
                for token in self.graph_parser.tokenizer.cut(param):
                    if token not in self._param_key_graph.value:
                        token_id = self._param_key_graph.insert_node(
                            0, key=META, value=token, self_loop=False, type=META)
                    else:
                        token_id = self._param_key_graph.value.index(token)
                    self._param_key_graph.insert_edges([token_id], [param_id])
            else:
                param_id = self._param_key_graph.value.index(param)
                self._param_key_graph.key[param_id] = KEY
                self._param_key_graph.type[param_id] = KEY

        self._edge_param_2d_idx = {
            x: i for i, x in enumerate(self._param_key_graph.value) if x in HeteroAttnConv.EDGE_PARAMS_2D}
        self._edge_param_1d_idx = {
            x: i for i, x in enumerate(self._param_key_graph.value) if x in HeteroAttnConv.EDGE_PARAMS_1D}
        self._node_param_2d_idx = {
            x: i for i, x in enumerate(self._param_key_graph.value)if x in HeteroAttnConv.NODE_PARAMS_2D}
        self._node_param_1d_idx = {
            x: i for i, x in enumerate(self._param_key_graph.value)
            if x in HeteroAttnConv.NODE_PARAMS_1D + [Target.TARGET_WEIGHT]}

        self.graph_parser.set_meta_mode(False)
        self._param_key_dgl_graph = dgl.graph(
            (self._param_key_graph.src, self._param_key_graph.dst), num_nodes=self._param_key_graph.num_nodes)
        self._param_key_dgl_graph.ndata[VALUE] = torch.tensor(
            list(map(self.graph_parser.encode, self._param_key_graph.value)), dtype=torch.long)

    def _init_meta_param(self):
        """生成元卷积层需要的参数"""
        self._param_key_dgl_graph = self._param_key_dgl_graph.to(self.device)
        with self._param_key_dgl_graph.local_scope():
            self._param_key_dgl_graph.ndata[FEATURE] = self.embedding(self._param_key_dgl_graph)
            self._param_key_dgl_graph.update_all(fn.copy_u(FEATURE, FEATURE), fn.sum(FEATURE, FEATURE))
            param_2d, param_1d = self.meta_learner(self._param_key_dgl_graph.ndata[FEATURE])

            def init_func(param: torch.Tensor, *idx_dicts: Dict[str, int]):
                for idx_dict in idx_dicts:
                    for key, idx in idx_dict.items():
                        self.register_buffer(key, param[idx], persistent=False)

            init_func(param_2d, self._node_param_2d_idx, self._edge_param_2d_idx)
            init_func(param_1d, self._node_param_1d_idx, self._edge_param_1d_idx)

    def eval(self):
        self._init_meta_param()
        return super().eval()

    def _init_param(self, meta_feats: torch.Tensor, meta_values: torch.Tensor):
        """
        生成卷积层需要的参数
        :param meta_feats: (num_blocks, num_nodes, d_model)
        :param meta_values: (num_blocks, num_edges, d_model)
        :return:
        """
        self._dynamic_params = {}
        node_weights, node_bias = self.meta_learner(meta_feats)
        edge_weights, edge_bias = self.meta_learner(meta_values)
        for key in self._edge_param_2d_idx:
            self._dynamic_params[key] = getattr(self, key).unsqueeze(0) + edge_weights
        for key in self._edge_param_1d_idx:
            self._dynamic_params[key] = getattr(self, key).unsqueeze(0) + edge_bias
        for key in self._node_param_2d_idx:
            self._dynamic_params[key] = getattr(self, key).unsqueeze(0) + node_weights
        for key in self._node_param_1d_idx:
            self._dynamic_params[key] = getattr(self, key).unsqueeze(0) + node_bias

    def _asign_param_to_graph(self, graph, i_layer=None):
        if i_layer is None:
            for key, param in self.named_buffers(recurse=False):
                if key in self._node_param_2d_idx or key in self._node_param_1d_idx:
                    graph.ndata[key] = param.repeat(graph.num_nodes(), *[1] * len(param.shape))
                elif key in self._edge_param_2d_idx or key in self._edge_param_1d_idx:
                    graph.edata[key] = param.repeat(graph.num_edges(), *[1] * len(param.shape))
        else:
            for key in self._edge_param_2d_idx:
                graph.edata[key] = self._dynamic_params[key][i_layer, graph.edata[META_EDGE_ID]]
            for key in self._edge_param_1d_idx:
                graph.edata[key] = self._dynamic_params[key][i_layer, graph.edata[META_EDGE_ID]]
            for key in self._node_param_2d_idx:
                graph.ndata[key] = self._dynamic_params[key][i_layer, graph.ndata[META_NODE_ID]]
            for key in self._node_param_1d_idx:
                graph.ndata[key] = self._dynamic_params[key][i_layer, graph.ndata[META_NODE_ID]]

    def forward_meta(self, meta_graph):
        if self.training:
            self._init_meta_param()

        # 生成元图的embedding
        meta_feats = [[] for _ in range(len(self.blocks)+1)]  # 多存了一层初始的embedding
        meta_keys = [[] for _ in range(len(self.blocks))]
        meta_values = [[] for _ in range(len(self.blocks))]
        meta_attn_weights = [[] for _ in range(len(self.blocks))]

        meta_graph = meta_graph.to(self.device)
        with meta_graph.local_scope():
            meta_feat = self.embedding(meta_graph)
            meta_feats[0].append(meta_feat)
            self._asign_param_to_graph(meta_graph)
            for i, block in enumerate(self.meta_blocks):
                for meta_conv in block:
                    meta_feat, meta_key, meta_value, meta_attn_weight = meta_conv(meta_graph, meta_feat)
                    meta_feats[i+1].append(meta_feat)  # 多存了一层初始的embedding
                    meta_keys[i].append(meta_key)
                    meta_values[i].append(meta_value)
                    meta_attn_weights[i].append(meta_attn_weight)
                if self.block_residual:
                    meta_feat = meta_feat + meta_feats[i][-1]

        self._init_param(
            meta_feats=torch.stack([x[-1] for x in meta_feats[1:]], dim=0),
            meta_values=torch.stack([x[-1] for x in meta_values], dim=0))
        return meta_feats, meta_keys, meta_values, meta_attn_weights

    def forward(self, graph: Union[dgl.DGLGraph, Graph], meta_graph: dgl.DGLGraph = None, details=False):
        if meta_graph is None:
            self.graph_parser.set_meta_mode(True)
            meta_graph, graph = self.graph_parser.graph_to_dgl(graph)
        meta_feats, meta_keys, meta_values, meta_attn_weights = self.forward_meta(meta_graph)
        output, feats, keys, values, attn_weights = super().forward(graph, details=True)
        if details:
            return output, meta_feats, meta_keys, meta_values, meta_attn_weights, \
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
        return '{}\nparam_key_graph={}'.format(super().extra_repr(), self._param_key_dgl_graph)
