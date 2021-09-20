import torch
import dgl
from ..globals import TYPE, WEIGHT, SEP, TARGET


class Target(torch.nn.Module):
    TARGET_WEIGHT = SEP.join([TARGET, WEIGHT])

    def forward(self, graph: dgl.DGLGraph, feat, target_type: torch.Tensor) -> torch.Tensor:
        """

        :param graph:
            ndata:
                READOUT: (num_nodes, emb_dim)
                TYPE: (num_nodes,)
        :param feat: (num_nodes, emb_dim)
        :param target_type: (n_type,)
        :return:
        """
        mask = (graph.ndata[TYPE].view(-1, 1) == target_type).any(-1)
        feat = feat[mask]
        weight = graph.ndata[self.TARGET_WEIGHT][mask].view(feat.shape)
        return (feat * weight).sum(-1)
