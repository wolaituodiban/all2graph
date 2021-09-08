import dgl
import dgl.function as fn
import torch


def edge_feed_forward(
        graph: dgl.DGLGraph,
        feat: torch.Tensor,
        u_weight: torch.Tensor,
        v_weight: torch.Tensor,
        e_weight: torch.Tensor,
        v_bias: torch.Tensor,
        u_bias: torch.Tensor,
        e_bias: torch.Tensor,
        activation: torch.nn.Module,
        dropout1: torch.nn.Module,
        dropout2: torch.nn.Module,
        norm: torch.nn.Module
) -> torch.Tensor:
    """
    u表示前置节点的特征向量，v表示后置节点的特征向量
    norm(dropout(dropout(activation(u * W_u + b_u + v * W_v + b_v)) * W_e + b_e) + u + v)
    返回边uv上的特征向量

    :param graph     : 图
    :param feat      : num_nodes * embedding_dim
    :param v_weight  : num_edges * embedding_dim ** 2
    :param u_weight  : num_edges * embedding_dim ** 2
    :param e_weight  : num_edges * embedding_dim ** 2
    :param v_bias    : num_edges * embedding_dim
    :param u_bias    : num_edges * embedding_dim
    :param e_bias    : num_edges * embedding_dim
    :param activation: 激活层
    :param dropout1  :
    :param dropout2  :
    :param norm      : 归一化层
    :return          : num_edges * embedding_dim
    """
    num_edges = graph.num_edges()
    with graph.local_scope():
        graph.ndata['feat'] = feat
        graph.edata['u_weight'] = u_weight
        graph.edata['v_weight'] = v_weight

        # 第一层全连接
        graph.apply_edges(fn.u_dot_e('feat', 'u_weight', 'u_feat'))
        graph.apply_edges(fn.v_dot_e('feat', 'v_weight', 'v_feat'))
        e_feat = graph.edata['u_feat'].view(num_edges, -1) + u_bias + graph.edata['v_feat'].view(num_edges, -1) + v_bias
        if activation is not None:
            e_feat = activation(e_feat)
        e_feat = dropout1(e_feat)

        # 第二层全连接
        e_feat = (e_feat.view(num_edges, -1, 1) * e_weight).sum(-1, keepdim=False) + e_bias
        graph.edata['e_feat'] = dropout2(e_feat)

        # add & norm
        graph.apply_edges(fn.u_add_e('feat', 'e_feat', 'e_feat'))
        graph.apply_edges(fn.v_add_e('feat', 'e_feat', 'e_feat'))
        output = graph.edata['e_feat'].view(num_edges, -1)
        if norm is not None:
            output = norm(output)
        return output


def node_feed_forward(
        feat: torch.Tensor,
        weight1: torch.Tensor,
        bias1: torch.Tensor,
        weight2: torch.Tensor,
        bias2: torch.Tensor,
        activation: torch.nn.Module,
        dropout1: torch.nn.Module,
        dropout2: torch.nn.Module,
        norm: torch.nn.Module
) -> torch.Tensor:
    """

    :param feat      : num_nodes * embedding_dim
    :param weight1   : num_nodes * embedding_dim ** 2
    :param bias1     : num_nodes * embedding_dim
    :param weight2   : num_nodes * embedding_dim ** 2
    :param bias2     : num_nodes * embedding_dim
    :param activation:
    :param dropout1  :
    :param dropout2  :
    :param norm      :
    :return          : num_nodes * embedding_dim
    """
    feat2 = (feat.view(*feat.shape, 1) * weight1).sum(-1, keepdim=True) + bias1
    feat2 = dropout1(activation(feat2))
    feat2 = (feat2 * weight2).sum(-1, keepdim=False) + bias2
    return norm(feat + dropout2(feat2))
