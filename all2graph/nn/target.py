import torch
import dgl
from ..globals import TYPE, WEIGHT, SEP, TARGET


class Target(torch.nn.Module):
    TARGET_WEIGHT = SEP.join([TARGET, WEIGHT])

    def __init__(self, targets: torch.Tensor):
        super().__init__()
        self.targets = targets

    def forward(self, graph: dgl.DGLGraph, feat) -> torch.Tensor:
        """

        :param graph:
            ndata:
                READOUT: (num_nodes, emb_dim)
                TYPE: (num_nodes,)
        :param feat: (num_nodes, emb_dim)
        :return:
        """
        mask = (graph.ndata[TYPE].view(-1, 1) == self.targets).any(-1)
        feat = feat[mask]
        weight = graph.ndata[self.TARGET_WEIGHT][mask].view(feat.shape)
        return (feat * weight).sum(-1)

    def extra_repr(self) -> str:
        return 'targets={}'.format(self.targets.tolist())
