from typing import Dict, List

import torch

from .functional import nodewise_linear
from .utils import _get_activation
from ..globals import SEP
from ..preserves import WEIGHT, TARGET, BIAS, HIDDEN


class FC(torch.nn.Module):
    TARGET_WEIGHT = SEP.join([TARGET, WEIGHT])
    TARGET_BIAS = SEP.join([TARGET, BIAS])
    TARGET_HIDDEN_WEIGHT = SEP.join([TARGET, HIDDEN, WEIGHT])
    TARGET_HIDDEN_BIAS = SEP.join([TARGET, HIDDEN, BIAS])

    def __init__(
            self, last_block_only=False, last_layer_only=False, share_block_param=False, bias=True,
            hidden_layers=1, hidden_bias=True
    ):
        """

        Args:
            last_block_only: 只使用最后一个block的特征
            last_layer_only: 只使用每个block最后一层的特征
            share_block_param: 同一个block公用
            bias: 是否使用bias
            hidden_layers: 隐藏层层数
            hidden_bias: 隐藏层bias
        """
        super().__init__()
        self.last_block_only = last_block_only
        self.last_layer_only = last_layer_only
        self.share_block_param = share_block_param
        self.bias = bias
        self.hidden_layers = hidden_layers
        self.hidden_bias = hidden_bias

    @property
    def parameter_names_0d(self):
        if self.bias:
            return [self.TARGET_BIAS]
        else:
            return []

    @property
    def target_hidden_weight(self):
        return [SEP.join([self.TARGET_HIDDEN_WEIGHT, str(i)]) for i in range(self.hidden_layers)]

    @property
    def target_hidden_bias(self):
        return [SEP.join([self.TARGET_HIDDEN_BIAS, str(i)]) for i in range(self.hidden_layers)]

    @property
    def parameter_names_1d(self):
        output = [self.TARGET_WEIGHT]
        if self.hidden_bias:
            output += self.target_hidden_bias
        return output

    @property
    def parameter_names_2d(self):
        return self.target_hidden_weight

    @property
    def parameter_names(self):
        return self.parameter_names_1d + self.parameter_names_0d + self.parameter_names_2d

    @property
    def node_parameter_names(self):
        return self.parameter_names

    def forward(
            self, feats: List[torch.Tensor], parameters: List[Dict[str, torch.Tensor]], mask: torch.Tensor,
            targets=List[str]):
        """

        Args:
            feats: 长度为num_blocks的list， 每个元素是tensor with shape(num_layers, num_nodes, emb_dim)，
                    ！注意每一个block的num_layers可能不一样！
            parameters: 长度为num_blocks的list， 每个元素是Dict of tensor,
                        TARGET_WEIGHT: (num_layers, num_nodes, emb_dim)
                        TARGET_BIAS: (num_layers, num_nodes, 1)
                        ！注意每一个block的num_layers可能不一样！但是与feats相同
            mask: (num_nodes, )
            targets:

        Returns:
            outputs: {target: tensor(num_samples, ) for target in self.targets}
            hidden_feats: [tensor(num_samples, dim)]
        """
        outputs = []
        hidden_feats = []
        for feat, param in zip(feats[-self.last_block_only:], parameters[-self.last_block_only:]):
            feat = feat[-self.last_layer_only:, mask]  # (num_layers, num_nodes, emb_dim)
            # 隐藏层
            for i in range(self.hidden_layers):
                num_layers, num_nodes, emb_dim = feat.shape
                feat = feat.view(num_layers * num_nodes, emb_dim)
                weight = param[self.target_hidden_weight[i]][-self.last_layer_only:, mask]
                weight = weight.view(-1, emb_dim, emb_dim)
                if self.hidden_bias:
                    bias = param[self.target_hidden_bias[i]][-self.last_layer_only:, mask]
                    bias = bias.view(-1, emb_dim)
                else:
                    bias = None
                feat = nodewise_linear(feat=feat, weight=weight, bias=bias)
                feat = feat.view(num_layers, num_nodes, emb_dim)
            # 隐藏层结果
            hidden_feats.append(feat)

            # 输出层
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
        return outputs, hidden_feats

    def extra_repr(self) -> str:
        msg = 'last_block_only={}, last_layer_only={}, share_block_param={}, bias={}, hidden_layers={}, hidden_bias={}'
        msg = msg.format(
            self.last_block_only, self.last_layer_only, self.share_block_param, self.bias, self.hidden_layers,
            self.hidden_bias
        )
        return msg
