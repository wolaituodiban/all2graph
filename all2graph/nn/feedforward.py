import torch

from .utils import Module, _get_activation, _get_norm


class FeedForward(Module):
    def __init__(self, num_feats, middle_feats=None, out_feats=None,
                 dropout=0, activation='prelu', norm='batch1d', norm_first=True, residual=True, pre=None):
        super().__init__()
        if pre is not None:
            self.pre = pre
        middle_feats = middle_feats or num_feats
        out_feats = out_feats or num_feats
        dropout = torch.nn.Dropout(dropout)
        linear1 = torch.nn.Linear(num_feats, middle_feats)
        norm1 = _get_norm(norm, middle_feats)
        activation = _get_activation(activation)
        linear2 = torch.nn.Linear(middle_feats, out_feats)
        if norm_first:
            layers = [dropout, linear1, norm1, activation, linear2]
        else:
            layers = [dropout, linear1, activation, norm1, linear2]
        self.layers = torch.nn.Sequential(*layers)
        if residual:
            self.norm = _get_norm(norm, out_feats)

    @property
    def device(self):
        return self.norm.weight

    def reset_parameters(self):
        if hasattr(self, 'pre') and hasattr(self.pre, 'reset_parameters'):
            self.pre.reset_parameters()
        for module in self.layers:
            if hasattr(module, 'reset_parameters'):
                module.reset_parameters()
        if hasattr(self, 'norm'):
            self.norm.reset_parameters()

    def forward(self, in_feats: torch.Tensor) -> torch.Tensor:
        if hasattr(self, 'pre'):
            in_feats = self.pre(in_feats)
        out_feats = self.layers(in_feats)
        if hasattr(self, 'norm'):
            out_feats = self.norm(out_feats + in_feats)
        return out_feats
