from typing import List, Union


import torch
from torch.utils.data import DataLoader
from toad.nn import Module

from .conv import Conv, Body, MockBody
from .embedding import NodeEmbedding, ValueEmbedding, MockValueEmbedding
from .meta import MetaLearner, MockMetaLearner
from .output import Output
from .utils import num_parameters
from ..graph import RawGraph, Graph
from all2graph.parsers.graph import RawGraphParser
from ..utils import progress_wrapper
from ..version import __version__


class _BaseEncoder(Module):
    """graph factorization machine"""
    def __init__(self, raw_graph_parser: RawGraphParser, d_model: int, emb_config: dict, num_weight: bool,
                 key_emb: bool, dropout: float, conv_config: dict, num_layers: List[int], share_layer: bool,
                 residual: bool, target_configs: dict):
        super().__init__()
        self.version = __version__
        self.raw_graph_parser = raw_graph_parser
        self.value_embedding = ValueEmbedding(
            num_embeddings=raw_graph_parser.num_strings, embedding_dim=d_model, **emb_config or {})
        self.node_embedding = NodeEmbedding(embedding_dim=d_model, num_weight=num_weight, key_bias=key_emb)

        conv_layer = Conv(normalized_shape=d_model, dropout=dropout, **conv_config or {})
        self.value_blocks = Body(
            num_layers=num_layers, conv_layer=conv_layer, share_layer=share_layer, residual=residual)

        self.output = Output(
            targets=self.raw_graph_parser.targets,
            symbols=self.raw_graph_parser.target_symbol,
            **target_configs or {})

        self.loss = None
        self._optimizer_config = {}

    @property
    def meta_value_embedding(self):
        return self.value_embedding

    @property
    def d_model(self):
        return self.value_embedding.embedding.weight.shape[1]

    @property
    def num_layers(self):
        return len(self.value_blocks)

    @property
    def num_keys(self):
        return self.raw_graph_parser.num_keys

    @property
    def num_etypes(self):
        return self.raw_graph_parser.num_etypes

    @property
    def device(self):
        return self.value_embedding.weight.device

    @property
    def node_dynamic_parameter_names(self):
        return list(set(self.node_embedding.node_dynamic_parameter_names
                        + self.value_blocks.node_dynamic_parameter_names
                        + self.output.node_dynamic_parameter_names + self.meta_blocks.node_dynamic_parameter_names))

    @property
    def edge_dynamic_parameter_names(self):
        return list(set(self.value_blocks.edge_dynamic_parameter_names + self.meta_blocks.edge_dynamic_parameter_names))

    @property
    def dynamic_parameter_names_2d(self):
        return list(set(self.meta_blocks.dynamic_parameter_names_2d + self.value_blocks.dynamic_parameter_names_2d))

    @property
    def dynamic_parameter_names_1d(self):
        return list(set(self.node_embedding.dynamic_parameter_names_1d + self.meta_blocks.dynamic_parameter_names_1d +
                        self.value_blocks.dynamic_parameter_names_1d + self.output.dynamic_parameter_names_1d))

    @property
    def dynamic_parameter_names_0d(self):
        return self.output.dynamic_parameter_names_0d

    @property
    def node_dynamic_parameter_names_2d(self):
        return list(set(self.node_dynamic_parameter_names).intersection(self.dynamic_parameter_names_2d))

    @property
    def node_dynamic_parameter_names_1d(self):
        return list(set(self.node_dynamic_parameter_names).intersection(self.dynamic_parameter_names_1d))

    @property
    def node_dynamic_parameter_names_0d(self):
        return list(set(self.node_dynamic_parameter_names).intersection(self.dynamic_parameter_names_0d))

    @property
    def edge_dynamic_parameter_names_2d(self):
        return list(set(self.edge_dynamic_parameter_names).intersection(self.dynamic_parameter_names_2d))

    @property
    def edge_dynamic_parameter_names_1d(self):
        return list(set(self.edge_dynamic_parameter_names).intersection(self.dynamic_parameter_names_1d))

    def set_loss(self, loss):
        self.loss = loss

    def set_optimizer_config(self, config=None):
        self._optimizer_config = dict(config or {})

    def optimizer(self):
        return torch.optim.AdamW(self.parameters(), **self._optimizer_config)

    def reset_parameters(self):
        self.value_embedding.reset_parameters()
        self.node_embedding.reset_parameters()
        self.meta_blocks.reset_parameters()
        self.value_blocks.reset_parameters()
        self.meta_learner.reset_parameters()

    def eval(self):
        with torch.no_grad():
            self.meta_learner.update_embedding(self.value_embedding)
        return super().eval()

    def forward(self, graph: Union[RawGraph, Graph], details=False):
        if self.training:
            self.meta_learner.update_embedding(self.value_embedding)

        if isinstance(graph, RawGraph):
            graph = self.raw_graph_parser.parse(graph)

        meta_graph = graph.meta_graph.to(self.device)
        meta_emb = self.meta_value_embedding(meta_graph)
        meta_conv_param = self.meta_learner(self.meta_blocks.dynamic_parameter_names)
        meta_feats, meta_keys, meta_values, meta_attn_weights = self.meta_blocks(
            graph=meta_graph, in_feat=meta_emb, parameters=[meta_conv_param] * len(self.meta_blocks))

        value_graph = graph.value_graph.to(self.device)
        value_emb = self.value_embedding(value_graph)
        node_emb_param = self.meta_learner(self.node_embedding.dynamic_parameter_names, feat=meta_emb)
        value_emb = self.node_embedding(
            value_emb, number=graph.number.to(self.device), parameters=node_emb_param, meta_node_id=graph.meta_node_id)
        value_conv_params = []
        for meta_feat, meta_value in zip(meta_feats, meta_values):
            value_conv_param = self.meta_learner(self.value_blocks.node_dynamic_parameter_names, meta_feat[-1])
            value_conv_param.update(self.meta_learner(self.value_blocks.edge_dynamic_parameter_names, meta_value[-1]))
            value_conv_params.append(value_conv_param)
        value_feats, value_keys, value_values, value_attn_weights = self.value_blocks(
            graph=value_graph, in_feat=value_emb, parameters=value_conv_params, meta_node_id=graph.meta_node_id,
            meta_edge_id=graph.meta_edge_id)

        output_params = [
            self.meta_learner(self.output.dynamic_parameter_names, feat=meta_feat[-1]) for meta_feat in meta_feats]
        outputs = self.output.forward(
            feats=value_feats, symbols=graph.symbol.to(self.device), meta_node_id=graph.meta_node_id,
            parameters=output_params)

        if details:
            return outputs, value_feats, value_keys, value_values, value_attn_weights, meta_feats, meta_keys,\
                   meta_values, meta_attn_weights
        else:
            return outputs

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
        s = 'num_parameters={}\nraw_graph_parser={}'
        return s.format(num_parameters(self), self.raw_graph_parser)


class Encoder(_BaseEncoder):
    def __init__(self, raw_graph_parser: RawGraphParser, d_model, nhead, num_latent, num_layers: List[int],
                 num_weight=True, key_emb=True, dropout=0.1, emb_config=None, conv_config=None,
                 share_layer=False, residual=True, target_configs=None, meta_norm=True):
        super().__init__(
            raw_graph_parser=raw_graph_parser, d_model=d_model, emb_config=emb_config or {}, num_weight=num_weight,
            key_emb=key_emb, dropout=dropout, conv_config=conv_config or {}, num_layers=num_layers,
            share_layer=share_layer, residual=residual, target_configs=target_configs)
        conv_layer = Conv(normalized_shape=d_model, dropout=dropout, **conv_config or {})
        self.meta_blocks = Body(
            num_layers=num_layers, conv_layer=conv_layer, share_layer=share_layer, residual=residual)
        self.meta_learner = MetaLearner(
            raw_graph_parser, d_model=d_model, num_latent=num_latent, param_shapes={
                (nhead, d_model // nhead, d_model): self.dynamic_parameter_names_2d,
                (nhead, d_model // nhead): self.dynamic_parameter_names_1d,
                (1,): self.dynamic_parameter_names_0d},
            dropout=dropout, norm=meta_norm)


class MockEncoder(_BaseEncoder):
    def __init__(self, raw_graph_parser: RawGraphParser, d_model, nhead, num_layers: List[int], num_weight=True,
                 key_emb=True, dropout=0.1, emb_config=None, conv_config=None, share_layer=False,
                 residual=True, target_configs=None):
        super().__init__(
            raw_graph_parser=raw_graph_parser, d_model=d_model, emb_config=emb_config or {}, num_weight=num_weight,
            key_emb=key_emb, dropout=dropout, conv_config=conv_config or {}, num_layers=num_layers,
            share_layer=share_layer, residual=residual, target_configs=target_configs)
        self.mock_value_embedding = MockValueEmbedding()
        self.meta_blocks = MockBody(num_layers=len(num_layers))
        self.meta_learner = MockMetaLearner(
            num_etypes=self.raw_graph_parser.num_etypes, num_ntypes=self.raw_graph_parser.num_keys,
            edge_param_shapes={
                (nhead, d_model // nhead, d_model): self.edge_dynamic_parameter_names_2d,
                (nhead, d_model // nhead): self.edge_dynamic_parameter_names_1d},
            node_param_shapes={
                (nhead, d_model // nhead, d_model): self.node_dynamic_parameter_names_2d,
                (nhead, d_model // nhead): self.node_dynamic_parameter_names_1d,
                (1, ): self.node_dynamic_parameter_names_0d},
        )

    @property
    def meta_value_embedding(self):
        return self.mock_value_embedding
