import torch.nn

import all2graph as ag
import matplotlib.pyplot as plt


def test_forward():
    raw_graph = ag.graph.RawGraph()
    raw_graph.add_kv_(0, 'a', 'b', True)
    raw_graph.add_kv_(0, 'a', 3, True)
    raw_graph.add_kv_(0, ('a', 'b'), 'b', True)
    raw_graph.add_kv_(0, ('a', 'b'), 'c', True)
    raw_graph.add_kv_(0, 'a', 'b', True)
    raw_graph.add_kv_(1, 'a', '0.23', True)
    raw_graph.add_kv_(0, ('a', 'b'), 'b', True)
    raw_graph.add_kv_(1, ('a', 'b'), 'c', True)
    raw_graph.add_kv_(2, ('a', 'b'), 'c', True)
    raw_graph.add_readouts_(['sdf', 'ge'], True)
    raw_graph.draw(key=True)
    plt.show()

    meta_info = raw_graph.meta_info()
    graph_parser = ag.GraphParser.from_data(meta_info)
    graph = graph_parser(raw_graph)

    d_model = 12
    module = ag.nn.Framework(
        token_emb=torch.nn.Embedding(graph_parser.num_tokens, d_model),
        number_emb=ag.nn.NumEmb(d_model),
        bottle_neck=ag.nn.BottleNeck(d_model),
        key_body=ag.nn.GATBody(d_model, 3, 2),
        value_body=ag.nn.GATBody(d_model, 3, 2),
        readout=ag.nn.Readout(d_model)
    )
    module.reset_parameters()
    print(module)
    pred = module(graph)
    print(pred)
    for k, v in pred.items():
        v.sum().backward()
        break


if __name__ == '__main__':
    test_forward()
