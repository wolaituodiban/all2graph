import numpy as np
import torch


def num_parameters(module: torch.nn.Module):
    parameters = {}
    for p in module.parameters():
        parameters[id(p)] = p
    return sum(map(lambda x: np.prod(x.shape), parameters.values()))
