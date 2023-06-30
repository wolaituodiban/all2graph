from copy import deepcopy

import dgl
import torch
import numpy as np
import pandas as pd

from .utils import Module
from .loss import DeepHitSingleLoss
from .block import Block
from ..graph import EventGraph
from ..parser import Parser

class Model(Module):
    def __init__(
        self,
        parser: Parser,
        d_model,
        nhead,
        num_tfm_layers,
        num_survival_periods,
        dim_feedforward=None,
        dropout=0,
        norm_first=True,
        unit=86400,
        epsilon=None
    ) -> None:
        super().__init__()
        if dim_feedforward is None:
            dim_feedforward = d_model
            
        self.parser = parser
        self.register_buffer('sqrt_d_model', torch.sqrt(torch.tensor(d_model, dtype=torch.float32)))

        self.embedding = torch.nn.Embedding(parser.num_embeddings, d_model)
        
        # lookup table编码器
        encoder_layer = torch.nn.TransformerEncoderLayer(
            d_model, nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            norm_first=norm_first
        )
        self.attr_encoder = torch.nn.TransformerEncoder(encoder_layer, num_tfm_layers)
        self.attr_norm = torch.nn.BatchNorm1d(d_model)
        # 事件编码器
        
        self.event_encoder = torch.nn.TransformerEncoder(encoder_layer, num_tfm_layers)
        self.event_norm = torch.nn.BatchNorm1d(d_model)
        self.event_type_norm = torch.nn.BatchNorm1d(d_model)
        
        # 时间编码器
        self.time_encoder = torch.nn.Sequential(
            torch.nn.BatchNorm1d(EventGraph.NUM_EDGE_FEAT),
            torch.nn.Linear(EventGraph.NUM_EDGE_FEAT, d_model),
            torch.nn.BatchNorm1d(d_model),
            torch.nn.PReLU()
        )
        
        # 因果图编码器
        block = Block(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward,
            dropout=dropout, norm_first=norm_first)
        self.causal_convs = dgl.nn.pytorch.Sequential(*[deepcopy(block) for _ in range(parser.num_layers)])
        
        # 生存分析器
        self.survival_conv = dgl.nn.pytorch.GATv2Conv(
            in_feats=d_model,
            out_feats=d_model//nhead,
            num_heads=nhead,
            feat_drop=dropout,
            attn_drop=dropout,
            residual=True,
            activation=torch.nn.PReLU(),
            allow_zero_in_degree=True
        )
        self.survival_norm = torch.nn.BatchNorm1d(d_model)
        self.survival_linear = torch.nn.Linear(d_model, num_survival_periods)
        self.survival_loss = DeepHitSingleLoss(unit, epsilon=epsilon)
        self.register_buffer('survival_periods', torch.arange(num_survival_periods, dtype=torch.float32)) 
        
        self.return_loss = True
        
        self._reset_parameters()
        
    def _reset_parameters(self):
        r"""Initiate parameters in the transformer model."""
        for p in self.parameters():
            if p.dim() > 1:
                torch.nn.init.xavier_uniform_(p)
                
    def add_tokens(self, new_tokens):
        # 保存老embedding参数
        old_embedding_state_dict = self.embedding.state_dict()
        d_model = old_embedding_state_dict['weight'].shape[1]
        
        # 将新token添加到字典中
        self.parser.add_tokens(new_tokens)
        num_new_tokens = self.parser.num_embeddings - old_embedding_state_dict['weight'].shape[0]
        
        # 创建新的embedding layer
        self.embedding = torch.nn.Embedding(self.parser.num_embeddings, d_model)
        
        # 用unknown的参数初始化新token的参数
        new_embedding_weight = old_embedding_state_dict['weight'][self.parser.unknown]
        new_embedding_weight = new_embedding_weight.repeat(num_new_tokens, 1)
        old_embedding_state_dict['weight'] = torch.cat([old_embedding_state_dict['weight'], new_embedding_weight], axis=0)
        self.embedding.load_state_dict(old_embedding_state_dict)
                
    def generate_square_subsequent_mask(self, inputs):
        neg_inf = torch.full((inputs.shape[1], inputs.shape[1]), float('-inf'), device=self.device)
        mask = torch.triu(neg_inf, diagonal=1)
        return mask

    def _attr_block(self, graph: EventGraph):
        lookup_mask = self.generate_square_subsequent_mask(graph.lookup_table)
        lookup_padding_mask = graph.lookup_table == self.parser.padding
        lookup_emb = self.embedding(graph.lookup_table) * self.sqrt_d_model
        lookup_emb = self.attr_encoder(lookup_emb, mask=lookup_mask, src_key_padding_mask=lookup_padding_mask)
        lookup_emb = self.attr_norm(lookup_emb[:, -1])
        return lookup_emb

    def _event_block(self, graph: EventGraph, lookup_emb):
        event_padding_mask = graph.events < 0
        event_feats = torch.nn.functional.embedding(graph.events.clip(0), lookup_emb)
        event_type_emb = self.event_type_norm(event_feats[:, 0])
        event_feats = self.event_encoder(event_feats, src_key_padding_mask=event_padding_mask)
        event_emb = self.event_norm(event_feats[:, -1])
        return event_type_emb, event_emb

    def _survival_block(self, graph: EventGraph, causal_feats, event_type_emb):
        survival_feats = self.survival_conv(graph.survival_graph, (causal_feats, event_type_emb))
        survival_feats = survival_feats.view(survival_feats.shape[0], -1)
        survival_feats = self.survival_norm(survival_feats)
        survival_probs = self.survival_linear(survival_feats).softmax(-1)

        output = {
            EventGraph.FEAT: survival_feats,
            'survival_prob': survival_probs,
            EventGraph.ARRIVAL_TIME: (survival_probs * self.survival_periods).sum(-1),
        }
        
        if self.return_loss:
            lower, upper = graph.survival_times
            mask = ~torch.isnan(graph.obs_timestamps)
            output['loss'] = self.survival_loss(survival_probs[mask], lower[mask], upper[mask])

        return output

    def forward(self, graph: EventGraph):
        if isinstance(graph, pd.DataFrame):
            graph = self.parser(graph)

        graph = graph.to(self.device, non_blocking=True)

        # lookup编码
        lookup_emb = self._attr_block(graph)

        # 事件embedding
        event_type_emb, event_emb = self._event_block(graph, lookup_emb)

        # 时间embedding
        edge_feats = self.time_encoder(graph.edge_feats)

        # 图卷积
        causal_feats, edge_emb = self.causal_convs(graph.causal_graph, event_emb, edge_feats)
        
        # 生存分析
        output = self._survival_block(graph, causal_feats=causal_feats, event_type_emb=event_type_emb)
        
        output.update({
            EventGraph.SAMPLE_ID: graph.sample_ids,
            EventGraph.SAMPLE_OBS_TIMESTAMP: graph.sample_obs_timestamps,
            EventGraph.OBS_TIMESTAMP: graph.obs_timestamps,
        })
        
        return output
    

class ModelV2(Model):
    def _survival_block(self, graph: EventGraph, causal_feats, event_type_emb):
        survival_feats = self.survival_conv(graph.survival_graph, (causal_feats, event_type_emb))
        survival_feats = survival_feats.view(survival_feats.shape[0], -1)        

        # v1先batchnorm，再indexing
        # v2先indexing，在batchnorm
        # 两者batchnrom的running_mean和running_var将不同
        mask = ~torch.isnan(graph.obs_timestamps)
        survival_feats = self.survival_norm(survival_feats[mask])
        survival_probs = self.survival_linear(survival_feats).softmax(-1)

        output = {
            EventGraph.FEAT: survival_feats,
            'survival_prob': survival_probs,
            EventGraph.ARRIVAL_TIME: (survival_probs * self.survival_periods).sum(-1),
        }
        
        if self.return_loss:
            lower, upper = graph.survival_times
            output['loss'] = self.survival_loss(survival_probs, lower[mask], upper[mask])

        return output
