from typing import Dict, List, Optional

import torch
import numpy as np

from .utils import Module


class DictLoss(Module):
    def __init__(self, torch_loss: torch.nn.Module, weights: Optional[Dict[str, float]] = None):
        """
        封装输入类型为dict的loss
        Args:
            torch_loss:
            weights:
        """
        super(DictLoss, self).__init__()
        self.torch_loss = torch_loss
        self.weights = weights or {}

    @property
    def device(self):
        raise NotImplementedError

    def forward(self, inputs: Dict[str, torch.Tensor], target: Dict[str, torch.Tensor]):
        if len(target) == 0:
            raise ValueError('get empty target! plz check your configs, especially RawGraphParser.targets')
        loss = 0
        weight_sum = 0
        for key, _target in target.items():
            if key not in inputs:
                continue
            _target = _target.to(inputs[key].device)
            mask = torch.bitwise_not(torch.isnan(_target))
            if mask.sum() == 0:
                continue
            weight = self.weights.get(key, 1)

            loss += weight * self.torch_loss(inputs[key][mask], _target[mask])
            weight_sum += weight

        return loss / weight_sum


class ListLoss(Module):
    def __init__(self, torch_loss: torch.nn.Module, weights: Optional[List[float]] = None):
        """
        封装输入类型为list的loss
        Args:
            torch_loss:
            weights:
        """
        super(ListLoss, self).__init__()
        self.torch_loss = torch_loss
        self.weights = weights

    @property
    def device(self):
        raise NotImplementedError

    def forward(self, inputs: List[torch.Tensor], target: List[torch.Tensor]):
        loss = 0
        weight_sum = 0
        for i, (_input, _target) in enumerate(zip(inputs, target)):
            if self.weights:
                weight = self.weights[i]
            else:
                weight = 1
            _target = _target.to(_input.device)
            mask = torch.bitwise_not(torch.isnan(_target))
            if mask.sum() == 0:
                continue
            loss += weight * self.torch_loss(_input[mask], _target[mask])
            weight_sum += weight

        return loss / weight_sum

class DeepHitSingleLoss(torch.nn.Module):
    def __init__(self, unit=1, epsilon=None):
        super().__init__()
        self.unit = unit
        if epsilon is not None:
            self.epsilon = epsilon
    
    def forward(self, pred, lower, upper):
        lower = lower.unsqueeze(-1) / self.unit
        upper = upper.unsqueeze(-1) / self.unit

        idx = torch.ones_like(pred).cumsum(-1) - 1
        lower_mask = idx >= lower.floor().clip(-np.inf, pred.shape[1]-1)
        upper_mask = idx <= upper.floor().clip(0, np.inf)
        mask = lower_mask & upper_mask
        
        prob = (pred * mask).sum(-1)
        if getattr(self, 'epsilon', None) is not None:
            prob = prob.clip(self.epsilon, 1)
        return -prob.log().mean()
    
    def extra_repr(self) -> str:
        return f'unit={self.unit}, epsilon={getattr(self, "epsilon", None)}'