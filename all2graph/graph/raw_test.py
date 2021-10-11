import os
import pandas as pd
from all2graph.json import JsonParser

path = os.path.dirname(__file__)
path = os.path.dirname(path)
path = os.path.dirname(path)
csv_path = os.path.join(path, 'test_data', 'MensShoePrices.csv')
node_df = pd.read_csv(csv_path, nrows=64)

parser = JsonParser(
    'json', flatten_dict=True, local_id_keys={'name'}, segment_value=True, self_loop=True,
    list_inner_degree=1
)
graph, global_index_mapper, local_index_mappers = parser.parse(
    node_df, progress_bar=True
)


def test():
    meta_graph, meta_node_id, meta_edge_id = graph.meta_graph()
    assert graph.key == [meta_graph.value[i] for i in meta_node_id]
    assert graph.component_id == [meta_graph.component_id[i] for i in meta_node_id]
    assert [graph.key[i] for i in graph.src] == [meta_graph.value[meta_graph.src[i]] for i in meta_edge_id]
    assert [graph.key[i] for i in graph.dst] == [meta_graph.value[meta_graph.dst[i]] for i in meta_edge_id]
    print(graph)


if __name__ == '__main__':
    test()
