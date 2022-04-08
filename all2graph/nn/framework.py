import torch

from .body import Body
from .bottle_neck import BottleNeck
from .embedding import NumEmb
from .head import Head
from .utils import Module
from ..graph import Graph


class Framework(Module):
    def __init__(self, str_emb: torch.nn.Embedding, key_emb, num_emb: NumEmb, bottle_neck: BottleNeck,
                 body: Body, head: Head = None, add_self_loop=True, to_bidirectied=False, to_simple=False,
                 seq_types=None):
        super().__init__()
        self.str_emb = str_emb
        self.key_emb = key_emb
        self.num_emb = num_emb
        self.bottle_neck = bottle_neck
        self.body = body
        self.head = head
        self.add_self_loop = add_self_loop
        self.to_bidirectied = to_bidirectied
        self.to_simple = to_simple
        self.seq_types = seq_types

    def reset_parameters(self):
        super().reset_parameters()

    def forward(self, graph: Graph, details=False):
        graph = graph.to(self.device, non_blocking=True)
        if self.add_self_loop:
            graph = graph.add_self_loop()
        if self.to_bidirectied:
            graph = graph.to_bidirectied(copy_ndata=True)
        if self.to_simple:
            graph = graph.to_simple(copy_ndata=True)

        # 计算key emb
        key_emb_ori = self.str_emb(graph.type_string)
        key_emb_ori = self.key_emb(key_emb_ori)
        # 兼容pytorch的recurrent layers和transformer layers
        if isinstance(key_emb_ori, tuple):
            key_emb_ori = key_emb_ori[0]
        key_emb_ori = key_emb_ori[:, -1]
        key_emb = key_emb_ori[graph.types]

        # 计算in_feats
        str_emb = self.str_emb(graph.strings)
        num_emb = self.num_emb(graph.numbers)
        bottle_neck = self.bottle_neck(key_emb, str_emb, num_emb)

        # 卷积
        ind1, ind2 = graph.seq2node
        ind1 = ind1.unsqueeze(-1).expand(-1, bottle_neck.shape[1])
        ind2 = ind2.unsqueeze(-1).expand(-1, bottle_neck.shape[1])
        ind3 = torch.arange(bottle_neck.shape[1], device=self.device).expand(graph.num_nodes, -1)
        if self.seq_types is None:
            seq_mask = None
        else:
            seq_mask = graph.seq_mask(self.seq_types)
        feats = self.body(
            graph.graph, bottle_neck, node2seq=graph.node2seq, seq2node=(ind1, ind2, ind3), seq_mask=seq_mask)

        # 输出
        readout_feats = feats[-1][graph.readout_mask]
        if self.head is None:
            output = readout_feats
        else:
            target_feats = {target: key_emb_ori[i] for target, i in graph.targets.items()}
            output = self.head(readout_feats, target_feats)
        if details:
            graph.ndata['key_emb'] = key_emb
            graph.ndata['str_emb'] = str_emb
            graph.ndata['num_emb'] = num_emb
            graph.ndata['bottle_neck'] = bottle_neck
            graph.ndata['feats'] = torch.stack(feats, dim=1)
            return output, graph
        else:
            return output
