import dgl
import dgl.function as fn
import torch


def edgewise_feedforward(
        graph: dgl.DGLGraph,
        feat: torch.Tensor,
        u_weight: torch.Tensor,
        v_weight: torch.Tensor,
        v_bias: torch.Tensor,
        u_bias: torch.Tensor,
        e_weight: torch.Tensor,
        e_bias: torch.Tensor,
        activation: torch.nn.Module,
        dropout1: torch.nn.Module,
        dropout2: torch.nn.Module,
        norm: torch.nn.Module
) -> torch.Tensor:
    """
    u表示前置节点的特征向量，v表示后置节点的特征向量
    norm(dropout(dropout(activation(u * W_u + b_u + v * W_v + b_v)) * W_e + b_e) + u + v)

    :param graph     : 图
    :param feat      : num_nodes * in_dim
    :param u_weight  : num_edges * nheads * mid_dim * in_dim
    :param v_weight  : num_edges * nheads * mid_dim * in_dim
    :param u_bias    : num_edges * nheads * mid_dim
    :param v_bias    : num_edges * nheads * mid_dim
    :param e_weight  : num_edges * nheads * mid_dim * out_dim
    :param e_bias    : num_edges * nheads * out_dim
    :param activation: 激活层
    :param dropout1  :
    :param dropout2  :
    :param norm      : 归一化层
    :return          : num_edges * nheads * out_dim
    """
    with graph.local_scope():
        graph.ndata['feat'] = feat
        graph.edata['u_weight'] = u_weight
        graph.edata['v_weight'] = v_weight
        # 第一层全连接
        graph.apply_edges(fn.u_dot_e('feat', 'u_weight', 'u_feat'))
        graph.apply_edges(fn.v_dot_e('feat', 'v_weight', 'v_feat'))
        u_feat = graph.edata['u_feat']
        v_feat = graph.edata['v_feat']
        e_feat = u_feat.view(u_feat.shape[:-1]) + u_bias + v_feat.view(v_feat.shape[:-1]) + v_bias
        if activation is not None:
            e_feat = activation(e_feat)
        e_feat = dropout1(e_feat)  # num_nodes * nhead * mid_dim

        # 第二层全连接
        e_feat = (e_feat.view(*e_feat.shape, 1) * e_weight).sum(-2, keepdim=False) + e_bias
        graph.edata['e_feat'] = dropout2(e_feat)  # num_nodes * nhead * out_dim

        # add & norm
        graph.ndata['feat'] = feat.view(graph.num_nodes(), *e_feat.shape[1:])
        graph.apply_edges(fn.u_add_e('feat', 'e_feat', 'e_feat'))
        graph.apply_edges(fn.v_add_e('feat', 'e_feat', 'e_feat'))
        output = graph.edata['e_feat']
        if norm is not None:
            output = norm(output)
        return output


def nodewise_feedforward(
        feat: torch.Tensor,
        w0: torch.Tensor,
        b0: torch.Tensor,
        w1: torch.Tensor,
        b1: torch.Tensor,
        activation: torch.nn.Module,
        dropout1: torch.nn.Module,
        dropout2: torch.nn.Module,
        norm: torch.nn.Module
) -> torch.Tensor:
    """
    u表示前置节点的特征向量，v表示后置节点的特征向量
    norm(dropout(dropout(activation(u * W_u + b_u + v * W_v + b_v)) * W_e + b_e) + u + v)

    :param feat      : num_nodes * num_layers * in_dim
    :param w0        : num_nodes * num_layers * nheads * mid_dim * in_dim
    :param b0        : num_nodes * num_layers * nheads * mid_dim
    :param w1        : num_nodes * num_layers * nheads * mid_dim * out_dim
    :param b1        : num_nodes * num_layers * nheads * out_dim
    :param activation: 激活层
    :param dropout1  :
    :param dropout2  :
    :param norm      : 归一化层
    :return          : num_nodes * num_layers * nheads * out_dim
    """
    # 第一层
    shape = [1] * len(w0.shape)
    shape[0], shape[1], shape[-1] = feat.shape[0], feat.shape[1], feat.shape[-1]
    output = (feat.view(shape) * w0).sum(-1) + b0
    # (num_nodes, num_layers, nheads, mid_dim)
    if activation is not None:
        output = activation(output)
    output = dropout1(output)
    # 第二层
    output = (output.view(*output.shape, 1) * w1).sum(-2) + b1  # (num_nodes, num_layers, nheads, out_dim)
    output = dropout2(output)
    # add & norm
    output += feat.view(output.shape)
    if norm is not None:
        output = norm(output)
    return output
