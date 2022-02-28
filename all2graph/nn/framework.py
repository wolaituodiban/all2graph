from typing import Dict

import torch

from .utils import Module
from ..graph import Graph


class Framework(Module):
    def __init__(self, token_emb, number_emb, bottle_neck, key_body, value_body, output):
        super().__init__()
        self.token_emb = token_emb
        self.number_emb = number_emb
        self.bottle_neck = bottle_neck
        self.key_body = key_body
        self.value_body = value_body
        self.output = output

    @property
    def device(self):
        return self.value_body.device

    def forward(self, graph: Graph) -> Dict[str, torch.Tensor]:
        key_token_emb = self.token_emb(graph.key_token)
        key_emb = self.key_body(graph, key_token_emb)
        value_token_emb = self.token_emb(graph.value_token)
        number_emb = self.number_emb(graph.number)
        in_feat = self.bottle_neck(key_emb, value_token_emb, number_emb)
        out_feat = self.value_body(graph, in_feat)
        return self.output(graph, out_feat)
