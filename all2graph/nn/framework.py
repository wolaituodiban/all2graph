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
                 batch_first=False):
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
        self.batch_first = batch_first

    def reset_parameters(self):
        super().reset_parameters()

    def forward(self, graph: Graph, details=False):
        if self.add_self_loop:
            graph = graph.add_self_loop()
        if self.to_bidirectied:
            graph = graph.to_bidirectied(copy_ndata=True)
        if self.to_simple:
            graph = graph.to_simple(copy_ndata=True)
        graph = graph.to(self.device, non_blocking=True)

        # 计算key emb
        if self.batch_first:
            key_tensor = graph.key_tensor
        else:
            key_tensor = graph.key_tensor.transpose(0, 1)
        key_emb_ori = self.str_emb(key_tensor)
        key_emb_ori = self.key_emb(key_emb_ori)
        # 兼容pytorch的recurrent layers和transformer layers
        if isinstance(key_emb_ori, tuple):
            key_emb_ori = key_emb_ori[0]
        if self.batch_first:
            key_emb_ori = key_emb_ori.mean(dim=1)
        else:
            key_emb_ori = key_emb_ori.mean(dim=0)
        key_emb = key_emb_ori[graph.keys]
        # print(key_emb.shape)
        str_emb = self.str_emb(graph.strings)
        num_emb = self.num_emb(graph.numbers)
        bottle_neck = self.bottle_neck(key_emb, str_emb, num_emb)
        # print(bottle_neck.shape)
        feats = self.body(graph.graph, in_feats=bottle_neck, seq_ids=graph.seq_ids)
        readout_feats = feats[-1][graph.readout_ids]
        if self.head is None:
            output = readout_feats
        else:
            target_feats = {target: key_emb_ori[graph.key_mapper[target]] for target in graph.targets}
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
