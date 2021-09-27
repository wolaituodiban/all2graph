import torch
import dgl

from ..globals import VALUE, NUMBER


class ValueEmbedding(torch.nn.Module):
    def __init__(self, embedding_dim, num_embeddings, **kwargs):
        super().__init__()
        self.embedding = torch.nn.Embedding(embedding_dim=embedding_dim, num_embeddings=num_embeddings, **kwargs)
        self.number_norm = torch.nn.BatchNorm1d(1)

    @property
    def device(self):
        return self.embedding.weight.device

    def forward(self, g: dgl.DGLGraph) -> torch.Tensor:
        output = self.embedding(g.ndata[VALUE])
        if NUMBER in g.ndata:
            mask = torch.isnan(g.ndata[NUMBER])
            mask = torch.bitwise_not(mask)
            output = torch.masked_fill(output, mask.view(-1, 1), 0)
            output[mask] += self.number_norm(g.ndata[NUMBER][mask].view(-1, 1))
        return output

    def reset_parameters(self):
        for module in self.children():
            if hasattr(module, 'reset_parameters'):
                module.reset_parameters()
