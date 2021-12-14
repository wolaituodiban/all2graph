import dgl
import dgl.function as fn
import torch


def edgewise_linear(
        graph: dgl.DGLGraph,
        feat: torch.Tensor,
        u_weight: torch.Tensor,
        v_weight: torch.Tensor,
        dropout: torch.nn.Module,
        v_bias: torch.Tensor = None,
        u_bias: torch.Tensor = None,
        norm: torch.nn.Module = None,
        activation: torch.nn.Module = None,
) -> torch.Tensor:
    """
    u表示前置节点的特征向量，v表示后置节点的特征向量
    activation(dropout(u) * W_u + b_u + dropout(v) * W_v + b_v)

    :param graph     : 图
    :param feat      : num_nodes * in_dim
    :param u_weight  : num_edges * nheads * out_dim * in_dim
    :param v_weight  : num_edges * nheads * out_dim * in_dim
    :param u_bias    : num_edges * nheads * out_dim
    :param v_bias    : num_edges * nheads * out_dim
    :param norm      : 激活层之前的归一化
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
        e_feat = graph.edata['u_feat'] + graph.edata['v_feat']
        e_feat = e_feat.view(e_feat.shape[:-1])
        if u_bias is not None:
            e_feat = e_feat + u_bias
        if v_bias is not None:
            e_feat = e_feat + v_bias
        if norm is not None:
            shape = e_feat.shape
            e_feat = norm(e_feat.view(*shape[:-2], -1)).view(shape)
        if activation is not None:
            e_feat = activation(e_feat)
        return e_feat


def nodewise_linear(
        feat: torch.Tensor,
        weight: torch.Tensor,
        bias: torch.Tensor = None,
        dropout: torch.nn.Module = None,
        norm: torch.nn.Module = None,
        activation: torch.nn.Module = None,
) -> torch.Tensor:
    """

    :param feat      : (num_nodes, in_dim)
    :param weight    : (num_nodes, *, in_dim)
    :param bias      : (num_nodes, *)
    :param norm      : 激活层之前的归一化
    :param activation: 激活层
    :param dropout   :
    :return          : (num_nodes, *)
    """
    if dropout is not None:
        feat = dropout(feat)
    shape = feat.shape[0], *[1] * (len(weight.shape) - len(feat.shape) - 1), feat.shape[-1], 1
    output = torch.matmul(weight, feat.view(shape)).view(weight.shape[:-1])
    if bias is not None:
        output += bias
    if norm is not None:
        shape = output.shape
        output = norm(output.view(*shape[:-2], -1)).view(shape)
    if activation is not None:
        output = activation(output)
    return output
