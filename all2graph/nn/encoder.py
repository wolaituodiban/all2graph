from typing import List, Tuple, Dict, Union

import dgl
import dgl.function as fn
import torch

from .conv import HeteroAttnConv
from .embedding import ValueEmbedding
from .meta import MetaLearner
from .target import Target
from .utils import num_parameters
from ..globals import FEATURE, SEP, VALUE, META_NODE_ID, META_EDGE_ID, TARGET, TYPE
from ..graph import Graph
from ..graph.parser import GraphParser


class UGFM(torch.nn.Module):
    """universal graph factorization machine"""
    TARGET_TYPE = SEP.join([TARGET, TYPE])

    def __init__(self, transer: GraphParser, d_model, num_latent, nhead, num_layers: List[Tuple[int, int]],
                 dropout=0.1, activation='relu', conv_residual=False, block_residual=True, share_conv=False, **kwargs):
        super(UGFM, self).__init__()
        self.transer = transer
        self.embedding = ValueEmbedding(num_embeddings=transer.num_strings, embedding_dim=d_model, **kwargs)
        self.meta_learner = MetaLearner(d_model, num_latent, nhead, d_model // nhead, dropout=dropout)
        self.blocks = torch.nn.ModuleList()
        self.target = Target()
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

        # 元参数信息
        self._init_meta()

    @property
    def device(self):
        return self.meta_learner.u.device

    def reset_parameters(self):
        for module in self.children():
            if hasattr(module, 'reset_parameters'):
                module.reset_parameters()

    def _init_meta(self):
        self.register_buffer(self.TARGET_TYPE, torch.tensor([self.transer.encode(TARGET)]))

        self._meta_node_param_1d_idx: Dict[str, int] = {}
        self._meta_node_param_2d_idx: Dict[str, int] = {}
        self._meta_edge_param_1d_idx: Dict[str, int] = {}
        self._meta_edge_param_2d_idx: Dict[str, int] = {}
        src, dst = [], []
        value = []

        def init_idx_and_graph(param_list: List[str], index_dict: Dict[str, int]):
            for name in param_list:
                index_dict[name] = len(value)
                value.append(name)
                for token in name.split(SEP):
                    src.append(len(value))
                    dst.append(index_dict[name])
                    value.append(token)

        init_idx_and_graph([Target.TARGET_WEIGHT], self._meta_node_param_1d_idx)
        init_idx_and_graph(HeteroAttnConv.NODE_PARAMS_1D, self._meta_node_param_1d_idx)
        init_idx_and_graph(HeteroAttnConv.NODE_PARAMS_2D, self._meta_node_param_2d_idx)
        init_idx_and_graph(HeteroAttnConv.EDGE_PARAMS_1D, self._meta_edge_param_1d_idx)
        init_idx_and_graph(HeteroAttnConv.EDGE_PARAMS_2D, self._meta_edge_param_2d_idx)

        self._meta_param_graph = dgl.graph((src, dst), num_nodes=len(value))
        self._meta_param_graph.ndata[VALUE] = torch.tensor(list(map(self.transer.encode, value)), dtype=torch.long)

    def _init_meta_param(self):
        self._meta_param_graph = self._meta_param_graph.to(self.device)
        with self._meta_param_graph.local_scope():
            self._meta_param_graph.ndata[FEATURE] = self.embedding(self._meta_param_graph)
            self._meta_param_graph.update_all(fn.copy_u(FEATURE, FEATURE), fn.sum(FEATURE, FEATURE))
            param_2d, param_1d = self.meta_learner(self._meta_param_graph.ndata[FEATURE])

            def init_param(param: torch.Tensor, *idx_dicts: Dict[str, int]):
                for idx_dict in idx_dicts:
                    for key, idx in idx_dict.items():
                        self.register_buffer(key, param[idx], persistent=False)

            init_param(param_2d, self._meta_node_param_2d_idx, self._meta_edge_param_2d_idx)
            init_param(param_1d, self._meta_node_param_1d_idx, self._meta_edge_param_1d_idx)

    def _asign_param_to_graph(self, graph, meta_node_feat=None, meta_edge_feat=None):
        for key, param in self.named_buffers(recurse=False):
            if key in self._meta_node_param_2d_idx or key in self._meta_node_param_1d_idx:
                graph.ndata[key] = param.repeat(graph.num_nodes(), *[1] * len(param.shape))
            elif key in self._meta_edge_param_2d_idx or key in self._meta_edge_param_1d_idx:
                graph.edata[key] = param.repeat(graph.num_edges(), *[1] * len(param.shape))

        if meta_node_feat is not None:
            node_param_2d, node_param_1d = self.meta_learner(meta_node_feat)
            for key in self._meta_node_param_2d_idx:
                graph.ndata[key] += node_param_2d[graph.ndata[META_NODE_ID]]
            for key in self._meta_node_param_1d_idx:
                graph.ndata[key] += node_param_1d[graph.ndata[META_NODE_ID]]

        if meta_edge_feat is not None:
            edge_param_2d, edge_param_1d = self.meta_learner(meta_edge_feat)
            for key in self._meta_edge_param_2d_idx:
                graph.edata[key] += edge_param_2d[graph.edata[META_EDGE_ID]]
            for key in self._meta_edge_param_1d_idx:
                graph.edata[key] += edge_param_1d[graph.edata[META_EDGE_ID]]

    def eval(self):
        self._init_meta_param()
        return super().eval()

    def forward(self, graph: Union[dgl.DGLGraph, Graph], meta_graph: dgl.DGLGraph = None):
        if self.training:
            self._init_meta_param()

        if meta_graph is None:
            meta_graph, graph = self.transer.graph_to_dgl(graph)
        meta_graph = meta_graph.to(self.device)
        graph = graph.to(self.device)

        output = 0
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
                # (num_components * num_targets, )
                if self.block_residual:
                    meta_node_feat += pre_meta_node_feat
                    node_feat += pre_node_feat
                target = self.target(graph, node_feat, self.get_buffer(self.TARGET_TYPE))
                target = target.view(-1, len(self.transer.targets))
                output += target
        output /= len(self.blocks)
        return output

    def extra_repr(self) -> str:
        return 'block_residual={}, num_parameters={}'.format(self.block_residual, num_parameters(self))
