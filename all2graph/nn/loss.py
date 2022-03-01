from typing import Dict, List
import torch

from .utils import Module


class DictLoss(Module):
    def __init__(self, torch_loss: torch.nn.Module, weights: Dict[str, float] = None):
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
    def __init__(self, torch_loss: torch.nn.Module, weights: List[float] = None):
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
