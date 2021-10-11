from typing import Dict, List

import torch

from ..globals import SEP
from ..preserves import WEIGHT, TARGET, BIAS


class Output(torch.nn.Module):
    TARGET_WEIGHT = SEP.join([TARGET, WEIGHT])
    TARGET_BIAS = SEP.join([TARGET, BIAS])

    def __init__(self, targets: List[str], symbols: List[int], last_block_only=False, last_layer_only=False):
        """

        :param targets: 目标的名称
        :param symbols: 目标的标记
        :param last_block_only: 只使用最后一个block的特征
        :param last_layer_only: 只使用每个block最后一层的特征
        """
        super().__init__()
        self.targets = targets
        self.register_buffer('symbols', torch.tensor(symbols, dtype=torch.long))
        self.last_block_only = last_block_only
        self.last_layer_only = last_layer_only

    @property
    def num_targets(self):
        return len(self.targets)

    @property
    def dynamic_parameter_names_0d(self):
        return [self.TARGET_BIAS]

    @property
    def dynamic_parameter_names_1d(self):
        return [self.TARGET_WEIGHT]

    @property
    def dynamic_parameter_names(self):
        return self.dynamic_parameter_names_1d + self.dynamic_parameter_names_0d

    @property
    def node_dynamic_parameter_names(self):
        return self.dynamic_parameter_names

    def forward(
            self, feats: List[List[torch.Tensor]], symbols: torch.Tensor, parameters: List[Dict[str, torch.Tensor]],
            meta_node_id: torch.Tensor) -> Dict[str, torch.Tensor]:
        """

        :param feats
        :param symbols:
        :param parameters:
        :param meta_node_id:
        :return:
        """

        outputs = []
        mask = (symbols.view(-1, 1) == self.symbols).any(-1)

        if self.last_block_only:
            feats = [feats[-1]]
            parameters = [parameters[-1]]
        for feat, param in zip(feats, parameters):
            masked_id = meta_node_id[mask]
            weight = param[self.TARGET_WEIGHT][masked_id]
            weight = weight.view(weight.shape[0], -1)
            bias = param[self.TARGET_BIAS][masked_id]

            if self.last_layer_only:
                feat = [feat[-1]]
            for layer_feat in feat:
                layer_feat = layer_feat[mask]
                output = (layer_feat * weight).sum(-1, keepdim=True) + bias
                outputs.append(output)
        outputs = torch.stack(outputs, dim=0)
        outputs = outputs.mean(dim=0)
        outputs = outputs.view(-1, self.num_targets)
        outputs = {target: outputs[:, i] for i, target in enumerate(self.targets)}
        return outputs

    def extra_repr(self) -> str:
        return 'targets={}, last_block_only={}, last_layer_only={}'.format(
            self.targets, self.last_block_only, self.last_layer_only)
