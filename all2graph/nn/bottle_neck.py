import torch

from .utils import Module, _get_activation
from ..graph import Graph
from ..globals import KEY, VALUE


class BottleNeck(Module):
    def __init__(self, d_model, dropout=0, activation='relu', norm_first=True):
        super().__init__()
        dropout = torch.nn.Dropout(dropout)
        linear = torch.nn.Linear(2 * d_model, d_model)
        norm = torch.nn.LayerNorm(d_model)
        activation = _get_activation(activation)
        if norm_first:
            self.layers = torch.nn.Sequential(dropout, linear, norm, activation)
        else:
            self.layers = torch.nn.Sequential(dropout, linear, activation, norm)

    @property
    def device(self):
        return self.layers[1].weight.device

    def reset_parameters(self):
        for layer in self.layers:
            if hasattr(layer, 'reset_parameters'):
                layer.reset_parameters()

    def forward(
            self, graph: Graph, key_feats: torch.Tensor, token_emb: torch.Tensor, num_emb: torch.Tensor
    ) -> torch.Tensor:
        key_feats = graph.push_key2value(key_feats)

        mask = torch.bitwise_not(torch.isnan(num_emb).any(-1))
        value_emb = token_emb.masked_fill(mask.unsqueeze(-1), 0)
        value_emb[mask] += num_emb[mask]

        kv_emb = torch.cat([key_feats, value_emb], dim=-1)
        return self.layers(kv_emb)
