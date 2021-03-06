import torch

from .utils import Module, _get_activation, _get_norm, reset_parameters


class BottleNeck(Module):
    def __init__(self, d_model, dropout=0, activation='prelu', norm='batch1d', norm_first=True):
        super().__init__()
        dropout = torch.nn.Dropout(dropout)
        linear = torch.nn.Linear(2 * d_model, d_model)
        norm = _get_norm(norm, d_model)
        activation = _get_activation(activation)
        if norm_first:
            self.layers = torch.nn.Sequential(dropout, linear, norm, activation)
        else:
            self.layers = torch.nn.Sequential(dropout, linear, activation, norm)

    def reset_parameters(self):
        for layer in self.layers:
            reset_parameters(layer)

    def forward(self, key_emb, str_emb, num_emb=None) -> torch.Tensor:
        if num_emb is not None:
            mask = torch.isnan(num_emb)
            num_emb = torch.masked_fill(num_emb, mask, 0)
            mask = torch.bitwise_not(mask)
            str_emb = torch.masked_fill(str_emb, mask, 0)
            str_emb = num_emb + str_emb
        return self.layers(torch.cat([key_emb, str_emb], dim=-1))
