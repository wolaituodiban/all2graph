from typing import List, Tuple

import torch

from ..globals import COMPONENT_ID


class Readout(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(
            self, feat: torch.Tensor, query: torch.Tensor, component_id: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """

        :param feat: tensor of shape (num_nodes, num_layers, embedding_dim)
        :param query: tensor of shape of (nheads, embedding_dim)
        :param component_id: tensor of long   , (num_nodes, )
        :return:
            feature of components: tensor of float32, (num_components, nheads, embedding_dim)
            attention_weight     : tensor of float32, (num_nodes, num_layers, nheads)
            component_id         : tensor of long   , (num_components, )
        """



        component_id = torch.unique(graph.ndata[COMPONENT_ID])
        component_mask = component_id.view(-1, 1) == graph.ndata[COMPONENT_ID].view(1, -1)