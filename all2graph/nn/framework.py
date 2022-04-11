from copy import deepcopy
from typing import Union, Dict

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
                 seq_types=None, num_heads=None):
        super().__init__()
        self.str_emb = str_emb
        self.key_emb = key_emb
        self.num_emb = num_emb
        self.bottle_neck = bottle_neck
        self.body = body
        if head is not None and num_heads is not None:
            self.head = torch.nn.ModuleList([deepcopy(head) for _ in range(num_heads)])
        else:
            self.head = head
        self.add_self_loop = add_self_loop
        self.to_bidirectied = to_bidirectied
        self.to_simple = to_simple
        self.seq_types = seq_types

    def reset_parameters(self):
        super().reset_parameters()

    def forward(self, graph: Graph, details=False) -> Union[Dict[str, torch.Tensor], Graph]:
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

        # 计算in_feats
        key_emb = key_emb_ori[graph.types]
        str_emb = self.str_emb(graph.strings)
        num_emb = self.num_emb(graph.numbers)
        bottle_neck = self.bottle_neck(key_emb, str_emb, num_emb)

        # 卷积
        feats = self.body(
            graph.graph, bottle_neck, node2seq=graph.node2seq, seq2node=graph.seq2node(bottle_neck.shape[1]),
            seq_mask=graph.seq_mask(self.seq_types))

        # 输出
        readout_mask = graph.readout_mask
        if self.head is None:  # 没有head时，输出embedding
            output = torch.stack([feat[readout_mask] for feat in feats], dim=1)
        else:
            target_feats = {target: key_emb_ori[graph.type_mapper[target]] for target in graph.targets}
            if isinstance(self.head, torch.nn.ModuleList):  # 多个head时，输出均值
                output = {target: [] for target in target_feats}
                for feat, head in zip(feats, self.head):
                    readout_feat = feat[readout_mask]
                    for target, pred in head(readout_feat, target_feats).items():
                        output[target].append(pred)
                output = {target: torch.stack(pred, dim=1).mean(dim=1) for target, pred in output.items()}
            else:  # 一个head时，输出head结果
                readout_feats = feats[-1][readout_mask]
                output = self.head(readout_feats, target_feats)
        if details:  # 将详细信息赋值给graph
            graph.ndata['key_emb'] = key_emb
            graph.ndata['str_emb'] = str_emb
            graph.ndata['num_emb'] = num_emb
            graph.ndata['bottle_neck'] = bottle_neck
            graph.ndata['feats'] = torch.stack(feats, dim=1)
            graph.output = output
            return graph
        else:
            return output
