import dgl
import dgl.function as fn
import torch


def edgewise_linear(
        graph: dgl.DGLGraph,
        feat: torch.Tensor,
        u_weight: torch.Tensor,
        v_weight: torch.Tensor,
        v_bias: torch.Tensor,
        u_bias: torch.Tensor,
        dropout: torch.nn.Module,
        activation: torch.nn.Module = None,
) -> torch.Tensor:
    """
    u表示前置节点的特征向量，v表示后置节点的特征向量
    norm(dropout(dropout(activation(u * W_u + b_u + v * W_v + b_v)) * W_e + b_e) + u + v)

    :param graph     : 图
    :param feat      : num_nodes * in_dim
    :param u_weight  : num_edges * nheads * out_dim * in_dim
    :param v_weight  : num_edges * nheads * out_dim * in_dim
    :param u_bias    : num_edges * nheads * out_dim
    :param v_bias    : num_edges * nheads * out_dim
    :param activation:
    :param dropout   :
    :return          : num_edges * nheads * out_dim
    """
    with graph.local_scope():
        graph.ndata['feat'] = dropout(feat)
        graph.edata['u_weight'] = u_weight
        graph.edata['v_weight'] = v_weight
        graph.apply_edges(fn.u_dot_e('feat', 'u_weight', 'u_feat'))
        graph.apply_edges(fn.v_dot_e('feat', 'v_weight', 'v_feat'))
        u_feat = graph.edata['u_feat']
        v_feat = graph.edata['v_feat']
        e_feat = u_feat.view(u_feat.shape[:-1]) + u_bias + v_feat.view(v_feat.shape[:-1]) + v_bias
        if activation is not None:
            e_feat = activation(e_feat)
        return e_feat


def nodewise_linear(
        feat: torch.Tensor,
        weight: torch.Tensor,
        bias: torch.Tensor,
        dropout: torch.nn.Module = None,
        activation: torch.nn.Module = None,
) -> torch.Tensor:
    """
    u表示前置节点的特征向量，v表示后置节点的特征向量
    norm(dropout(dropout(activation(u * W_u + b_u + v * W_v + b_v)) * W_e + b_e) + u + v)

    :param feat      : num_nodes * in_dim
    :param weight    : num_nodes * out_dim * in_dim
    :param bias      : num_nodes * out_dim
    :param activation: 激活层
    :param dropout   :
    :return          : num_nodes * out_dim
    """
    if dropout is not None:
        feat = dropout(feat)
    output = (feat.view(*feat.shape[:-1], 1, feat.shape[-1]) * weight).sum(-1) + bias
    if activation is not None:
        output = activation(output)
    return output
