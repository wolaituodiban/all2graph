from typing import Union, Dict, Tuple

import torch

from .body import Body
from .bottle_neck import BottleNeck
from .embedding import NumEmb
from .head import Head
from .utils import Module
from ..graph import Graph


class Framework(Module):
    def __init__(self, str_emb: torch.nn.Embedding, key_emb, num_emb: NumEmb, bottle_neck: BottleNeck,
                 body: Body, head: Head = None, num_featmaps=0, add_self_loop=True, to_bidirectied=False, to_simple=False,
                 seq_types=None, seq_degree: Tuple[int, int] = None):
        super().__init__()
        self.str_emb = str_emb
        self.key_emb = key_emb
        self.num_emb = num_emb
        self.bottle_neck = bottle_neck
        self.body = body
        self.head = head
        self.num_featmaps = num_featmaps
        self.add_self_loop = add_self_loop
        self.to_bidirectied = to_bidirectied
        self.to_simple = to_simple
        self.seq_types = seq_types
        self.seq_degree = seq_degree

    def reset_parameters(self):
        super().reset_parameters()

    def forward(self, graph: Graph, details=False) -> Union[Dict[str, torch.Tensor], torch.Tensor, Graph]:
        graph = graph.to(self.device, non_blocking=True)
        if self.add_self_loop:
            graph = graph.add_self_loop()
        if self.to_bidirectied:
            graph = graph.to_bidirectied(copy_ndata=True)
        if self.seq_degree:
            graph = graph.add_edges_by_seq(*self.seq_degree)
        if self.to_simple:
            graph = graph.to_simple(copy_ndata=True)

        # 计算key emb
        key_emb_ori = self.str_emb.forward(graph.type_string)
        key_emb_ori = self.key_emb.forward(key_emb_ori)
        # 兼容pytorch的recurrent layers和transformer layers
        if isinstance(key_emb_ori, tuple):
            key_emb_ori = key_emb_ori[0]
        key_emb_ori = key_emb_ori[:, -1]

        # 计算in_feats
        key_emb = key_emb_ori[graph.types]
        str_emb = self.str_emb.forward(graph.strings)
        num_emb = self.num_emb.forward(graph.numbers)
        bottle_neck = self.bottle_neck.forward(key_emb, str_emb, num_emb)

        # 卷积
        feats = self.body.forward(
            graph.graph, bottle_neck, node2seq=graph.node2seq, seq2node=graph.seq2node(bottle_neck.shape[1]),
            seq_mask=graph.seq_mask(self.seq_types))
        feats = feats[-self.num_featmaps:]
        if len(feats) > 1:
            feats = torch.cat(feats, dim=1)
        else:
            feats = feats[0]

        # 输出
        readout_feats = feats[graph.readout_mask]
        if self.head is None:  # 没有head时，输出embedding
            target_feats = None
            output = readout_feats
        else:
            target_feats = {target: key_emb_ori[graph.type_mapper[target]] for target in graph.targets}
            output = self.head.forward(readout_feats, target_feats)
        if details:  # 将详细信息赋值给graph
            graph.ndata['key_emb'] = key_emb
            graph.ndata['str_emb'] = str_emb
            graph.ndata['num_emb'] = num_emb
            graph.ndata['bottle_neck'] = bottle_neck
            graph.ndata['feats'] = torch.stack(feats, dim=1)
            graph.output = output
            graph.target_feats = target_feats
            return graph
        else:
            return output
