import math
import torch
from torch.nn.functional import linear

from .functional import nodewise_linear
from .utils import num_parameters


class MetaLearner(torch.nn.Module):
    """parameter projection"""
    def __init__(self, dim, num_latent, *inner_shape, dropout=0):
        super().__init__()
        self.a = torch.nn.Parameter(torch.Tensor(num_latent, dim))
        self.u = torch.nn.Parameter(torch.Tensor(1, num_latent, *inner_shape, dim))
        self.dropout = torch.nn.Dropout(dropout)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        torch.nn.init.kaiming_uniform_(self.a, a=math.sqrt(5))
        torch.nn.init.kaiming_uniform_(self.u, a=math.sqrt(5))

    def forward(self, emb: torch.Tensor) -> (torch.Tensor, torch.Tensor):
        """
        W = EAU
        B = dot(E, W)
        :param emb: (, dim)
        :return:
            weight: (, *, dim)
            bias  : (, *)
        """
        e_mul_a = linear(emb, self.a)  # (, num_latent)
        e_mul_a = e_mul_a.view(*e_mul_a.shape, *[1] * (len(self.u.shape) - len(e_mul_a.shape)))
        weight = (e_mul_a * self.u).sum(1)  # (, *, dim)
        bias = nodewise_linear(emb, weight, dropout=self.dropout)
        return weight, bias

    def extra_repr(self) -> str:
        return 'dim={}, num_latent={}, inner_shape={}, num_parameters={}'.format(
            self.a.shape[1], self.a.shape[0], tuple(self.u.shape[2:-1]), num_parameters(self))
