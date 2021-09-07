import dgl
import dgl.function as fn
import torch


def edge_feed_forward(
        graph: dgl.DGLGraph,
        feat: torch.Tensor,
        u_weight: torch.Tensor,
        u_bias: torch.Tensor,
        v_weight: torch.Tensor,
        v_bias: torch.Tensor,
        activation: torch.nn.Module,
        e_weight: torch.Tensor,
        e_bias: torch.Tensor,
        norm: torch.nn.Module
) -> torch.Tensor:
    """
    u表示前置节点的特征向量，v表示后置节点的特征向量
    norm(activation(u * W_e + b_e + v * W_n + b_n) * W_e + b_e + u + v)
    返回边uv上的特征向量

    :param graph     : 图
    :param feat      : num_nodes * embedding_dim
    :param v_weight  : num_edges * embedding_dim ** 2
    :param v_bias    : num_edges * embedding_dim
    :param u_weight  : num_edges * embedding_dim ** 2
    :param u_bias    : num_edges * embedding_dim
    :param activation: 激活层
    :param e_weight  : num_edges * embedding_dim ** 2
    :param e_bias    : num_edges * embedding_dim
    :param norm      : 归一化层
    :return          : num_edges * embedding_dim
    """
    num_nodes = graph.num_nodes()
    num_edges = graph.num_edges()
    with graph.local_scope():
        graph.ndata['feat'] = feat
        graph.edata['u_weight'] = u_weight
        graph.edata['v_weight'] =
        graph.apply_edges(fn.)

        graph.ndata['weight'] = v_weight
        graph.ndata['bias'] = v_bias
        graph.apply_nodes()

        graph.apply_edges(fn.u_dot_e('feat', ATTENTION_KEY_WEIGHT, KEY))
        graph.edata[KEY] = graph.edata[KEY].view(graph.num_edges(), -1)
        graph.edata[KEY] += graph.edata[ATTENTION_KEY_BIAS] + graph.edata[FEATURE]
        graph.apply_edges(fn.e_add_u(KEY, KEY, KEY))