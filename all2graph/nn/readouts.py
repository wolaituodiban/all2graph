import dgl
import numpy as np
import torch

from .functional import nodewise_feedforward
from ..globals import COMPONENT_ID, QUERY, \
    NODE_KEY_WEIGHT_1, NODE_KEY_BIAS_1, NODE_KEY_WEIGHT_2, NODE_KEY_BIAS_2, \
    NODE_VALUE_WEIGHT_1, NODE_VALUE_BIAS_1, NODE_VALUE_WEIGHT_2, NODE_VALUE_BIAS_2


class Readout(torch.nn.Module):
    def __init__(self, normalized_shape, activation='relu', dropout=0.1):
        super().__init__()
        if activation == 'gelu':
            activation = torch.nn.GELU()
        elif activation == 'relu':
            activation = torch.nn.ReLU()
        self.key_activation = activation
        self.value_activation = activation

        self.key_dropout1 = torch.nn.Dropout(dropout)
        self.key_dropout2 = torch.nn.Dropout(dropout)
        self.value_dropout1 = torch.nn.Dropout(dropout)
        self.value_dropout2 = torch.nn.Dropout(dropout)
        self.attn_dropout = torch.nn.Dropout(dropout)

        if normalized_shape is not None:
            self.key_norm = torch.nn.LayerNorm(normalized_shape)
            self.value_norm = torch.nn.LayerNorm(normalized_shape)
            self.out_norm = torch.nn.LayerNorm(normalized_shape)
        else:
            self.key_norm = None
            self.value_norm = None
            self.out_norm = None

    def forward(self, graph: dgl.DGLGraph, feat: torch.Tensor) -> (torch.Tensor, torch.Tensor, torch.Tensor):
        """
        :param graph:
            ndata:
                QUERY            : (, num_layers, nheads, embedding_dim // nheads)

                NODE_KEY_WEIGHT_1  : (, num_layers, nheads, feedforward_dim, embedding_dim )
                NODE_KEY_BIAS_1    : (, num_layers, nheads, feedforward_dim)

                NODE_KEY_WEIGHT_2  : (, num_layers, nheads, feedforward_dim, embedding_dim // nheads)
                NODE_KEY_BIAS_2    : (, num_layers, nheads, embedding_dim // nheads)

                NODE_VALUE_WEIGHT_1  : (, num_layers, nheads, feedforward_dim, embedding_dim )
                NODE_VALUE_BIAS_1    : (, num_layers, nheads, feedforward_dim)

                NODE_VALUE_WEIGHT_2  : (, num_layers, nheads, feedforward_dim, embedding_dim // nheads)
                NODE_VALUE_BIAS_2    : (, num_layers, nheads, embedding_dim // nheads)
        :param feat: (num_nodes, num_layers, embedding_dim)
        :return:
            FEATURE  : (num_components, embedding_dim)
            ATTENTION: (num_components, num_nodes, num_layser, nheads)
            COMPONENT_ID: (num_compoents, )
        """
        # 计算key
        key = nodewise_feedforward(
            feat=feat, w0=graph.ndata[NODE_KEY_WEIGHT_1], b0=graph.ndata[NODE_KEY_BIAS_1],
            w1=graph.ndata[NODE_KEY_WEIGHT_2], b1=graph.ndata[NODE_KEY_BIAS_2], activation=self.key_activation,
            dropout1=self.key_dropout2, dropout2=self.key_dropout2, norm=self.key_norm
        )  # (num_nodes, num_layers, nheads, embedding_dim // nheads)

        # 计算attentnion
        component_id = graph.ndata[COMPONENT_ID]  # (num_nodes, )
        unique_component_id = torch.unique(component_id)
        attention = (key * graph.ndata[QUERY]).sum(-1)  # (num_nodes, num_layers, nheads)
        attention = attention.repeat(unique_component_id.shape[0], *[1] * len(attention.shape))
        # 计算mask
        component_mask = unique_component_id.view(-1, 1) != component_id.view(1, -1)  # (num_components, num_nodes)
        component_mask = component_mask.view(*component_mask.shape, 1, 1)
        attention = torch.masked_fill(attention, component_mask, float('-inf'))
        # softmax
        attention = torch.softmax(attention.view(attention.shape[0], -1, attention.shape[-1]), dim=-2)
        attention = attention.view(attention.shape[0], *key.shape[:3])
        attention = self.attn_dropout(attention)  # (num_components, num_nodes, num_layers, nheads)

        # 计算value
        value = nodewise_feedforward(
            feat=feat, w0=graph.ndata[NODE_VALUE_WEIGHT_1], b0=graph.ndata[NODE_VALUE_BIAS_1],
            w1=graph.ndata[NODE_VALUE_WEIGHT_2], b1=graph.ndata[NODE_VALUE_BIAS_2],
            activation=self.key_activation, dropout1=self.key_dropout2, dropout2=self.key_dropout2, norm=self.key_norm
        )  # (num_nodes, num_layers, nheads, embedding_dim // nheads)

        # sum
        output = attention.view(*attention.shape, 1) * value.view(1, *value.shape)
        # (num_components, num_nodes, num_layers, nheads, embedding_dim // nheads)
        output = output.sum([1, 2])
        # add & norm
        if self.out_norm is not None:
            output = self.out_norm(output)
        return output.view(output.shape[0], -1), attention, unique_component_id
