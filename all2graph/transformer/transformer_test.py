import os
import json
import time
import pandas as pd
from all2graph import MetaGraph
from all2graph.json import JsonResolver
from all2graph.transformer import Transformer


def test():
    path = os.path.dirname(__file__)
    path = os.path.dirname(path)
    path = os.path.dirname(path)
    path = os.path.join(path, 'test_data', 'meta_graph.json')
    with open(path, 'r') as file:
        meta_graph = MetaGraph.from_json(json.load(file))
    Transformer.from_meta_graph(
        meta_graph, min_df=0.01, max_df=0.95, top_k=100, top_method='max_tfidf', name_segmentation=False
    )
    trans = Transformer.from_meta_graph(
        meta_graph, name_segmentation=False
    )

    path = os.path.dirname(__file__)
    path = os.path.dirname(path)
    path = os.path.dirname(path)
    csv_path = os.path.join(path, 'test_data', 'MensShoePrices.csv')
    df = pd.read_csv(csv_path, nrows=64)

    resolver = JsonResolver(
        root_name='json', flatten_dict=True, local_index_names={'name'}, segmentation=True
    )
    graph, global_index_mapper, local_index_mappers = resolver.resolve(
        list(map(json.loads, df.json)), progress_bar=True
    )
    start_time = time.time()
    dgl_meta_graph, dgl_graph = trans.graph_to_dgl(graph)
    print('graph_to_dgl', time.time() - start_time)
    print(dgl_meta_graph)
    print(dgl_graph)
    rc_graph = trans.graph_from_dgl(dgl_meta_graph, dgl_graph)

    assert graph.component_ids == rc_graph.component_ids
    assert graph.preds == rc_graph.preds
    assert graph.succs == rc_graph.succs
    assert rc_graph.names == graph.names
    values = pd.Series(graph.values)
    rc_values = pd.Series(rc_graph.values)
    numbers = pd.to_numeric(values, errors='coerce')
    str_mask = values.apply(lambda x: isinstance(x, str) and x in trans.string_mapper) & numbers.isna()
    assert all(values[str_mask] == rc_values[str_mask])


if __name__ == '__main__':
    test()
