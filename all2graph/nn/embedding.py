import torch
from ..globals import SEP
from ..preserves import NUMBER, WEIGHT, KEY, BIAS


class NodeEmbedding(torch.nn.Module):
    NUMBER_WEIGHT = SEP.join([NUMBER, WEIGHT])
    KEY_BIAS = SEP.join([KEY, BIAS])

    def __init__(self, embedding_dim, num_weight: bool, key_bias: bool):
        super().__init__()
        self.num_weight = num_weight
        if self.num_weight:
            self.num_norm = torch.nn.BatchNorm1d(embedding_dim)
        else:
            self.num_norm = torch.nn.BatchNorm1d(1)
        self.key_bias = key_bias

    @property
    def node_dynamic_parameter_names(self):
        return self.dynamic_parameter_names_1d

    @property
    def dynamic_parameter_names_1d(self):
        output = []
        if self.num_weight:
            output.append(self.NUMBER_WEIGHT)
        if self.key_bias:
            output.append(self.KEY_BIAS)
        return output

    @property
    def dynamic_parameter_names(self):
        return self.dynamic_parameter_names_1d

    @property
    def device(self):
        return self.embedding.weight.device

    def forward(self, feat, number, parameters) -> torch.Tensor:
        output = feat
        mask = torch.isnan(number)
        mask = torch.bitwise_not(mask)
        output = torch.masked_fill(output, mask.view(-1, 1), 0)
        number = number[mask].view(-1, 1)
        if self.num_weight and number.shape[0] > 0:
            num_weight = parameters[self.NUMBER_WEIGHT][mask]
            number = number * num_weight.view(num_weight.shape[0], -1)
        output[mask] += self.num_norm(number)
        if self.key_bias:
            key_bias = parameters[self.KEY_BIAS]
            output += key_bias.view(output.shape)
        return output

    def reset_parameters(self):
        for module in self.children():
            if hasattr(module, 'reset_parameters'):
                module.reset_parameters()

    def extra_repr(self) -> str:
        return 'num_weight={}, key_bias={}'.format(self.num_weight, self.key_bias)
