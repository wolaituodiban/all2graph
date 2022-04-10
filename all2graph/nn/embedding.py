import math

import numpy as np
import torch
from .utils import _get_activation, Module, _get_norm


class NumEmb(Module):
    """
    参考torch.nn.Embedding文档，将数值型特征也变成相同的形状。
    实际上就是将输入的张量扩展一个为1的维度之后，加上一个没有常数项的全连接层
    """
    def __init__(self, emb_dim, bias=True, activation='prelu', norm='batch1d'):
        super(NumEmb, self).__init__()
        self.weight = torch.nn.Parameter(torch.empty(emb_dim))
        if bias:
            self.bias = torch.nn.Parameter(torch.zeros(emb_dim))
        else:
            self.bias = None
        self.activation = _get_activation(activation)
        self.norm = _get_norm(norm, emb_dim)
        self.reset_parameters()

    def reset_parameters(self):
        bound = 1 / math.sqrt(self.weight.shape[0])
        torch.nn.init.uniform_(self.weight, -bound, bound)
        if self.bias is not None:
            torch.nn.init.zeros_(self.bias)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        mask = torch.isnan(inputs)
        output = torch.masked_fill(inputs, mask, 0)
        output = output.unsqueeze(-1) * self.weight
        if self.bias is not None:
            output = output + self.bias
        output = self.norm(output)
        if self.activation:
            output = self.activation(output)
        return torch.masked_fill(output, mask.unsqueeze(-1), np.nan)

    def extra_repr(self):
        return 'emb_dim={}, bias={}'.format(
            self.weight.shape[0], self.bias is not None
        )
