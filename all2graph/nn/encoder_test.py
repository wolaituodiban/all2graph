import torch

import all2graph
from all2graph import Graph, PRESERVED_WORDS, Timer, MetaInfo
from all2graph.graph.parser import GraphParser
from all2graph.nn import GFMEncoder, UGFMEncoder, num_parameters


def test_gfm():
    print('PRESERVED_WORDS:', PRESERVED_WORDS)
    graph = Graph(component_id=[0, 0], key=['readout', 'meta haha'], value=['value', 'node'],
                  type=['readout', 'value'], src=[0, 1, 1], dst=[0, 0, 1])

    meta_info = MetaInfo.from_data(graph)
    graph_parser = GraphParser.from_data(meta_info, targets=['a', 'b'])
    d_model = 8
    nhead = 2
    model1 = GFMEncoder(graph_parser, d_model=d_model, nhead=nhead, num_layers=[5], node_bias=True)
    print(model1.eval())
    with Timer('cpu forward'):
        out = model1(graph, details=True)

    if torch.cuda.is_available():
        model1 = model1.cuda()
        with Timer('gpu forward'):
            model1(graph, details=True)
        model1.train()
        with Timer('gpu forward'):
            model1(graph, details=True)
        with Timer('gpu forward'):
            out = model1(graph, details=True)
    # for n, o in zip(['output', 'feats', 'keys', 'values', 'attn_weights'], out):
    #     print(n)
    #     print(o)
    out[0]['a'].mean().backward()
    assert len(out[0]) == graph_parser.num_targets
    for v in out[0].values():
        assert v.shape == (graph.num_components, )


def test_ugfm():
    graph = Graph(component_id=[0, 0], key=['key haha', 'meta haha'], value=['value', 'node'],
                  type=['readout', 'value'], src=[0, 1, 1], dst=[0, 0, 1])

    meta_info = MetaInfo.from_data(graph)
    graph_parser = GraphParser.from_data(meta_info, targets=['a', 'b'], tokenizer=all2graph.default_tokenizer)

    d_model = 8
    num_latent = 4
    nhead = 2
    num_layers = [3, 1]
    num_meta_layers = [1, 2]
    model1 = UGFMEncoder(graph_parser, d_model=d_model, num_latent=num_latent, nhead=nhead, num_layers=num_layers,
                         share_conv=True, edge_bias=True, node_bias=True, num_meta_layers=num_meta_layers,
                         key_activation='relu', value_activation='relu', node_norm=True, key_norm=True, value_norm=True,
                         conv_residual=True)
    model2 = UGFMEncoder(graph_parser, d_model=d_model, num_latent=num_latent, nhead=nhead, num_layers=num_layers,
                         share_conv=False, edge_bias=True, node_bias=True, num_meta_layers=num_meta_layers,
                         key_activation='relu', value_activation='relu', node_norm=True, key_norm=True, value_norm=True,
                         conv_residual=True)
    assert num_parameters(model1) < num_parameters(model2)
    print(model1.eval())
    with torch.no_grad():
        for k, v in model1.named_buffers(recurse=False):
            print(k, v.shape, v.mean().numpy(), v.std().numpy())
            if len(v.shape) == 3:
                assert (v != 0).all()
            elif len(v.shape) == 2:
                assert (v == 0).all()
            else:
                raise AssertionError

    with Timer('cpu forward'):
        out = model1(graph, details=True)

    if torch.cuda.is_available():
        model1 = model1.cuda()
        with Timer('gpu forward'):
            model1(graph, details=True)
        model1.train()
        with Timer('gpu forward'):
            model1(graph, details=True)
        with Timer('gpu forward'):
            out = model1(graph, details=True)
    # for n, o in zip(['output', 'meta_feats', 'meta_keys', 'meta_values', 'meta_attn_weights', 'feats', 'keys',
    #                  'values', 'attn_weights'], out):
    #     print(n)
    #     print(o)
    out[0]['a'].mean().backward()
    assert len(out[0]) == graph_parser.num_targets
    for v in out[0].values():
        assert v.shape == (graph.num_components, )

    print(model1._param_key_dgl_graph.ndata)
    print(model1._param_key_dgl_graph.edges())


if __name__ == '__main__':
    test_gfm()
    test_ugfm()
