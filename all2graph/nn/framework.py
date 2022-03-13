from typing import Dict, Union

import torch

from .bottle_neck import BottleNeck
from .body import Body
from .embedding import NumEmb
from .readout import Readout
from .utils import Module
from ..graph import Graph
from ..globals import KEY, VALUE


class Framework(Module):
    def __init__(self, token_emb: torch.nn.Embedding, number_emb: NumEmb, bottle_neck: BottleNeck, key_body: Body,
                 value_body: Body, readout: Readout):
        super().__init__()
        self.register_buffer('_device_tracer', torch.ones(1))
        self.token_emb = token_emb
        self.key_body = key_body
        self.number_emb = number_emb
        self.bottle_neck = bottle_neck
        self.value_body = value_body
        self.readout = readout

    @property
    def device(self):
        return self._device_tracer.device

    def reset_parameters(self):
        super().reset_parameters()

    def forward(
            self, graph: Graph, details=False, ret_graph_emb=False
    ) -> Union[Graph, torch.Tensor, Dict[str, torch.Tensor]]:
        graph = graph.to(self.device, non_blocking=True)
        key_emb = self.token_emb(graph.key_token)
        key_feats = self.key_body(graph.key_graph, in_feats=key_emb)
        token_emb = self.token_emb(graph.value_token)
        num_emb = self.number_emb(graph.number)
        bottle_neck = self.bottle_neck(graph.push_key2value(key_feats[-1]), token_emb, num_emb)
        value_feats = self.value_body(graph.value_graph, in_feats=bottle_neck)
        if details:
            graph.nodes[KEY].data['emb'] = key_emb
            graph.nodes[KEY].data['feats'] = torch.stack(key_feats, dim=1)
            graph.nodes[VALUE].data['token_emb'] = token_emb
            graph.nodes[VALUE].data['num_emb'] = num_emb
            graph.nodes[VALUE].data['bottle_neck'] = bottle_neck
            graph.nodes[VALUE].data['feats'] = torch.stack(value_feats, dim=1)
            for ntype in graph.readout_types:
                graph.nodes[ntype].data['key_feats'] = graph.push_key2readout(key_feats[-1], ntype)
                graph.nodes[ntype].data['value_feats'] = graph.push_value2readout(value_feats[-1], ntype)
                if self.readout is not None:
                    graph.nodes[ntype].data['readout'] = self.readout(
                        key_feats=graph.nodes[ntype].data['key_feats'],
                        value_feats=graph.nodes[ntype].data['value_feats']
                    )
            return graph
        elif ret_graph_emb:
            return graph.push_value2readout(value_feats[-1], graph.readout_types[0])
        else:
            output = {}
            for ntype in graph.readout_types:
                output[ntype] = self.readout(
                    key_feats=graph.push_key2readout(key_feats[-1], ntype),
                    value_feats=graph.push_value2readout(value_feats[-1], ntype)
                )
            return output
