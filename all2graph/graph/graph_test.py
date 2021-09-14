import os
import json
import pandas as pd
from all2graph.json import JsonParser

path = os.path.dirname(__file__)
path = os.path.dirname(path)
path = os.path.dirname(path)
csv_path = os.path.join(path, 'test_data', 'MensShoePrices.csv')
node_df = pd.read_csv(csv_path, nrows=64)

parser = JsonParser(
    'json', flatten_dict=True, local_index_names={'name'}, segment_value=True, self_loop=True,
    list_inner_degree=1
)
graph, global_index_mapper, local_index_mappers = parser.parse(
    node_df, progress_bar=True
)


def test():
    meta_node_ids, meta_node_id_mapper, meta_node_component_ids, meta_node_names = graph.meta_node_info()
    assert graph.key == [meta_node_names[i] for i in meta_node_ids]
    assert graph.component_id == [meta_node_component_ids[i] for i in meta_node_ids]

    meta_edge_ids, pred_meta_node_ids, succ_meta_node_ids = graph.meta_edge_info(meta_node_id_mapper)
    assert [graph.key[i] for i in graph.src] == [meta_node_names[pred_meta_node_ids[i]] for i in meta_edge_ids]
    assert [graph.key[i] for i in graph.dst] == [meta_node_names[succ_meta_node_ids[i]] for i in meta_edge_ids]


if __name__ == '__main__':
    test()
