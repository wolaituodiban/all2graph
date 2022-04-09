import torch

from .utils import Module, _get_activation, _get_norm


class BottleNeck(Module):
    def __init__(self, d_model, num_inputs, dropout=0, activation='relu', norm='batch1d', norm_first=True):
        super().__init__()
        self.pre_norm = torch.nn.ModuleList(
            [_get_norm(norm, d_model) for _ in range(num_inputs)]
        )
        dropout = torch.nn.Dropout(dropout)
        linear = torch.nn.Linear(num_inputs * d_model, d_model)
        norm = _get_norm(norm, d_model)
        activation = _get_activation(activation)
        if norm_first:
            self.layers = torch.nn.Sequential(dropout, linear, norm, activation)
        else:
            self.layers = torch.nn.Sequential(dropout, linear, activation, norm)

    def reset_parameters(self):
        for layer in self.layers:
            if hasattr(layer, 'reset_parameters'):
                layer.reset_parameters()

    def forward(self, *inputs: torch.Tensor) -> torch.Tensor:
        inputs = [norm(x) for norm, x in zip(self.pre_norm, inputs)]
        kv_emb = torch.cat(inputs, dim=-1)
        return self.layers(kv_emb)
