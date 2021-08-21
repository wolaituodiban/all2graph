import os
import json
from all2graph import Transformer, MetaGraph


def test():
    path = os.path.dirname(__file__)
    path = os.path.dirname(path)
    path = os.path.dirname(path)
    path = os.path.join(path, 'test_data', 'meta_graph.json')
    with open(path, 'r') as file:
        meta_graph = MetaGraph.from_json(json.load(file))
    trans = Transformer(meta_graph)
    print(len(trans.string_mapper))


if __name__ == '__main__':
    test()
