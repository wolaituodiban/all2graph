from typing import Dict, List

import torch

from ..globals import SEP
from ..preserves import WEIGHT, TARGET, BIAS


class FC(torch.nn.Module):
    TARGET_WEIGHT = SEP.join([TARGET, WEIGHT])
    TARGET_BIAS = SEP.join([TARGET, BIAS])

    def __init__(
            self, last_block_only=False, last_layer_only=False, share_block_param=False, bias=True
    ):
        """

        :param last_block_only: 只使用最后一个block的特征
        :param last_layer_only: 只使用每个block最后一层的特征
        :param share_block_param: 同一个block公用
        :param bias: 是否使用bias
        """
        super().__init__()
        self.last_block_only = last_block_only
        self.last_layer_only = last_layer_only
        self.share_block_param = share_block_param
        self.bias = bias

    @property
    def num_targets(self):
        return len(self.targets)

    @property
    def dynamic_parameter_names_0d(self):
        if self.bias:
            return [self.TARGET_BIAS]
        else:
            return []

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
            self, feats: List[torch.Tensor], parameters: List[Dict[str, torch.Tensor]], mask: torch.Tensor,
            targets=List[str]) -> Dict[str, torch.Tensor]:
        """

        :param feats: 长度为num_blocks的list， 每个元素是tensor with shape(num_layers, num_nodes, emb_dim)，
                      ！注意每一个block的num_layers可能不一样！
        :param parameters: 长度为num_blocks的list， 每个元素是Dict of tensor,
                           TARGET_WEIGHT: (num_layers, num_nodes, emb_dim)
                           TARGET_BIAS: (num_layers, num_nodes, 1)
                           ！注意每一个block的num_layers可能不一样！但是与feats相同
        :param mask: (num_nodes, )
        :param targets:
        :return:
            {target: tensor(num_samples, ) for target in self.targets}
        """

        outputs = []

        for feat, param in zip(feats[-self.last_block_only:], parameters[-self.last_block_only:]):
            feat = feat[-self.last_layer_only:, mask]  # (num_layers, num_nodes, emb_dim)
            weight = param[self.TARGET_WEIGHT][-self.last_layer_only:, mask]  # (num_layers, num_nodes, emb_dim)
            weight = weight.view(feat.shape)  # 兼容性
            if self.share_block_param:
                weight = weight[[-1]]
            output = (feat * weight).sum(dim=-1, keepdim=True)
            if self.bias:
                bias = param[self.TARGET_BIAS][-self.last_layer_only:, mask]  # (num_layers, num_nodes, 1)
                if self.share_block_param:
                    bias = bias[[-1]]
                output += bias
            output = output.mean(dim=0)  # (num_nodes, 1)
            outputs.append(output)
        outputs = torch.stack(outputs, dim=0)
        outputs = outputs.mean(dim=0)
        outputs = outputs.view(-1, len(targets))
        outputs = {target: outputs[:, i] for i, target in enumerate(targets)}
        return outputs

    def extra_repr(self) -> str:
        return 'last_block_only={}, last_layer_only={}, share_block_param={}, bias={}'.format(
            self.last_block_only, self.last_layer_only, self.share_block_param, self.bias)
