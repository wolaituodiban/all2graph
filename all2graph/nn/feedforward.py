import torch

from .utils import Module, _get_activation


class FeedForward(Module):
    def __init__(self, num_feats, dropout=0, activation='relu', norm_first=True):
        super().__init__()
        dropout = torch.nn.Dropout(dropout)
        linear1 = torch.nn.Linear(num_feats, num_feats)
        norm1 = torch.nn.LayerNorm(num_feats)
        activation = _get_activation(activation)
        linear2 = torch.nn.Linear(num_feats, num_feats)
        if norm_first:
            self.layers = torch.nn.Sequential(dropout, linear1, norm1, activation, linear2)
        else:
            self.layers = torch.nn.Sequential(dropout, linear1, activation, norm1, linear2)
        self.norm = torch.nn.LayerNorm(num_feats)

    @property
    def device(self):
        return self.norm.weight

    def reset_parameters(self):
        for module in self.layers:
            if hasattr(module, 'reset_parameters'):
                module.reset_parameters()
        self.norm.reset_parameters()

    def forward(self, in_feats: torch.Tensor) -> torch.Tensor:
        out_feats = self.layers(in_feats)
        return self.norm(out_feats + in_feats)
