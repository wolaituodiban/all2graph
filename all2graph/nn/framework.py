from typing import Dict

import torch

from .utils import Module
from ..graph import Graph


class Framework(Module):
    def __init__(self, token_emb, number_emb, bottle_neck, key_body, value_body, readout):
        super().__init__()
        self.register_buffer('_device_tracer', torch.ones(1))
        self.token_emb = token_emb
        self.number_emb = number_emb
        self.bottle_neck = bottle_neck
        self.key_body = key_body
        self.value_body = value_body
        self.readout = readout

    @property
    def device(self):
        return self._device_tracer.device

    def reset_parameters(self):
        super().reset_parameters()

    def forward(self, graph: Graph) -> Dict[str, torch.Tensor]:
        graph = graph.to(self.device, non_blocking=True)
        key_emb = self.token_emb(graph.key_token)
        key_feats = self.key_body(graph.key_graph, in_feats=key_emb)
        token_emb = self.token_emb(graph.value_token)
        num_emb = self.number_emb(graph.number)
        value_feats = self.bottle_neck(graph, key_feats=key_feats, token_emb=token_emb, num_emb=num_emb)
        value_feats = self.value_body(graph.value_graph, in_feats=value_feats)
        return self.readout(graph, value_feats=value_feats, key_feats=key_feats)
