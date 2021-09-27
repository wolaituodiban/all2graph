import math
import torch
from torch.nn.functional import linear

from .utils import num_parameters


class MetaLearner(torch.nn.Module):
    """parameter projection"""
    def __init__(self, dim, num_latent, *inner_shape, dropout=0, gain=1):
        super().__init__()
        self.a = torch.nn.Parameter(torch.Tensor(num_latent, dim))
        self.u = torch.nn.Parameter(torch.Tensor(1, num_latent, *inner_shape, dim))
        self.v = torch.nn.Parameter(torch.Tensor(1, *[1] * len(inner_shape), dim))
        self.dropout = torch.nn.Dropout(dropout)
        self.norm = torch.nn.LayerNorm(dim)
        # 根据kaiming initialization 的理论，Var[W] = gain ** 2 / fan_mode
        self.register_buffer('scale', torch.tensor(gain/math.sqrt(dim)))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        torch.nn.init.kaiming_uniform_(self.a)
        torch.nn.init.kaiming_uniform_(self.u)
        torch.nn.init.constant_(self.v, 0)

    def forward(self, emb: torch.Tensor) -> (torch.Tensor, torch.Tensor):
        """
        W = EAU
        B = EAUV
        :param emb: (, dim)
        :return:
            weight: (, *, dim)
            bias  : (, *)
        """
        emb = self.dropout(emb)
        e_mul_a = linear(emb, self.a)  # (, num_latent)
        e_mul_a = e_mul_a.view(*e_mul_a.shape, *[1] * (len(self.u.shape) - len(e_mul_a.shape)))
        weight = (e_mul_a * self.u).sum(1)  # (, *, dim)
        # 根据kaiming initialization 的理论，Var[W] = gain ** 2 / fan_mode
        # 因此，此处要乘上一个系数
        weight = self.norm(weight) * self.scale
        bias = (weight * self.v).sum(-1)
        return weight, bias

    def extra_repr(self) -> str:
        return 'dim={}, num_latent={}, inner_shape={}, num_parameters={}'.format(
            self.a.shape[1], self.a.shape[0], tuple(self.u.shape[2:-1]), num_parameters(self))
