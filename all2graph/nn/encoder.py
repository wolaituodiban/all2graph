from typing import List, Tuple, Dict, Union

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
from ..globals import FEATURE, SEP, VALUE, META_NODE_ID, META_EDGE_ID, TARGET
from ..graph import Graph
from ..graph.parser import GraphParser
from ..utils import progress_wrapper


class UGFMEncoder(Module):
    """universal graph factorization machine"""
    def __init__(self, graph_parser: GraphParser, d_model, num_latent, nhead, num_layers: List[Tuple[int, int]],
                 dropout=0.1, activation='relu', conv_residual=False, block_residual=True, share_conv=False, **kwargs):
        super(UGFMEncoder, self).__init__()
        self.graph_parser = graph_parser
        self.embedding = ValueEmbedding(num_embeddings=graph_parser.num_strings, embedding_dim=d_model, **kwargs)
        self.meta_learner = MetaLearner(d_model, num_latent, nhead, d_model // nhead, dropout=dropout)
        self.blocks = torch.nn.ModuleList()
        self.target = Target(targets=torch.tensor([self.graph_parser.encode(TARGET)]))
        for n1, n2 in num_layers:
            if share_conv:
                meta_conv = torch.nn.ModuleList(
                    [HeteroAttnConv(d_model, dropout=dropout, activation=activation, residual=conv_residual)] * n1)
                conv = torch.nn.ModuleList(
                    [HeteroAttnConv(d_model, dropout=dropout, activation=activation, residual=conv_residual)] * n2)
            else:
                meta_conv = torch.nn.ModuleList(
                    [HeteroAttnConv(d_model, dropout=dropout, activation=activation, residual=conv_residual)
                     for _ in range(n1)])
                conv = torch.nn.ModuleList(
                    [HeteroAttnConv(d_model, dropout=dropout, activation=activation, residual=conv_residual)
                     for _ in range(n2)])
            block = torch.nn.ModuleDict({'meta_conv': meta_conv, 'conv': conv})
            self.blocks.append(block)
        self.block_residual = block_residual
        self.loss = None

        # 元参数信息
        self._init_param_key_graph()

    def set_loss(self, loss):
        self.loss = loss

    @property
    def device(self):
        return self.meta_learner.u.device

    def reset_parameters(self):
        for module in self.children():
            if hasattr(module, 'reset_parameters'):
                module.reset_parameters()

    def _init_param_key_graph(self):
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

    def _asign_param_to_graph(self, graph, meta_node_feat=None, meta_edge_feat=None):
        for key, param in self.named_buffers(recurse=False):
            if key in self._node_param_2d_idx or key in self._node_param_1d_idx:
                graph.ndata[key] = param.repeat(graph.num_nodes(), *[1] * len(param.shape))
            elif key in self._edge_param_2d_idx or key in self._edge_param_1d_idx:
                graph.edata[key] = param.repeat(graph.num_edges(), *[1] * len(param.shape))

        if meta_node_feat is not None:
            node_param_2d, node_param_1d = self.meta_learner(meta_node_feat)
            for key in self._node_param_2d_idx:
                graph.ndata[key] += node_param_2d[graph.ndata[META_NODE_ID]]
            for key in self._node_param_1d_idx:
                graph.ndata[key] += node_param_1d[graph.ndata[META_NODE_ID]]

        if meta_edge_feat is not None:
            edge_param_2d, edge_param_1d = self.meta_learner(meta_edge_feat)
            for key in self._edge_param_2d_idx:
                graph.edata[key] += edge_param_2d[graph.edata[META_EDGE_ID]]
            for key in self._edge_param_1d_idx:
                graph.edata[key] += edge_param_1d[graph.edata[META_EDGE_ID]]

    def eval(self):
        self._init_base_param()
        return super().eval()

    def forward(self, graph: Union[dgl.DGLGraph, Graph], meta_graph: dgl.DGLGraph = None):
        if self.training:
            self._init_base_param()

        if meta_graph is None:
            meta_graph, graph = self.graph_parser.graph_to_dgl(graph)
        meta_graph = meta_graph.to(self.device)
        graph = graph.to(self.device)

        output = []
        with meta_graph.local_scope(), graph.local_scope():
            meta_node_feat = self.embedding(meta_graph)
            node_feat = self.embedding(graph)
            self._asign_param_to_graph(meta_graph)
            for block in self.blocks:
                pre_meta_node_feat = meta_node_feat
                pre_node_feat = node_feat
                for meta_conv in block['meta_conv']:
                    meta_node_feat, meta_edge_feat, meta_attn_weight = meta_conv(meta_graph, meta_node_feat)
                self._asign_param_to_graph(graph, meta_node_feat=meta_node_feat, meta_edge_feat=meta_edge_feat)
                for conv in block['conv']:
                    node_feat, edge_feat, attn_weight = conv(graph, node_feat)
                    output.append(self.target(graph, node_feat))  # (num_components * num_targets, )
                if self.block_residual:
                    meta_node_feat = meta_node_feat + pre_meta_node_feat
                    node_feat = node_feat + pre_node_feat
        output = torch.stack(output, -1)
        output = torch.mean(output, -1)
        output = output.view(-1, len(self.graph_parser.targets))
        output = {target: output[:, i] for i, target in enumerate(self.graph_parser.targets)}
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
        return self.loss(labels, pred)

    def extra_repr(self) -> str:
        return 'block_residual={}, num_parameters={}'.format(self.block_residual, num_parameters(self))
