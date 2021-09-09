from typing import List, Tuple

import dgl
import torch

from ..globals import COMPONENT_ID


class Readout(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(
            self, graph: dgl.DGLGraph, feats: List[torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor, List[torch.Tensor]]:
        """

        :param graph:
        :param feats: List of tensors of shape [num_nodes * embedding_dim]
        :return:
            feture of components: tensor of float32, num_components * embedding_dim
            component_id        : tensor of long   , num_components
            attention_weight    : list of tensors of float32, num_components * nheads
        """
        component_id = torch.unique(graph.ndata[COMPONENT_ID])