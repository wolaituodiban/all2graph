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
    trans = Transformer.from_meta_graph(
        meta_graph, min_df=0.01, max_df=0.99, top_k=100, top_method='max_tfidf', split_name=True
    )
    print(len(trans.string_mapper))


if __name__ == '__main__':
    test()
