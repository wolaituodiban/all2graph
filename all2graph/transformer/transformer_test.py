import os
import json
import pandas as pd
from all2graph import MetaGraph
from all2graph.json import JsonResolver
from all2graph.transformer import Transformer
from all2graph.utils import Timer


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

    with Timer('graph_to_dgl'):
        dgl_meta_graph, dgl_graph = trans.graph_to_dgl(graph)
    print(dgl_meta_graph)
    print(dgl_graph)

    # 验证_gen_dgl_meta_graph的正确性
    rc_meta_node_df = pd.DataFrame({k: v.numpy() for k, v in dgl_meta_graph.ndata.items()})
    rc_meta_node_df['name'] = rc_meta_node_df['name'].map(trans.reverse_string_mapper)
    pred, succ = dgl_meta_graph.edges()
    rc_meta_edge_df = pd.DataFrame({'pred_meta_node_id': pred, 'succ_meta_node_id': succ})
    rc_meta_edge_df['component_id'] = rc_meta_node_df['component_id'].iloc[rc_meta_edge_df['pred_meta_node_id']].values
    assert all(rc_meta_edge_df['component_id'].values
               == rc_meta_node_df['component_id'].iloc[rc_meta_edge_df['succ_meta_node_id']].values)
    rc_meta_edge_df['pred_name'] = rc_meta_node_df['name'].iloc[rc_meta_edge_df['pred_meta_node_id']].values
    rc_meta_edge_df['succ_name'] = rc_meta_node_df['name'].iloc[rc_meta_edge_df['succ_meta_node_id']].values

    meta_edge_df = graph.meta_edge_df()
    assert all(rc_meta_edge_df['pred_name'].values == meta_edge_df['pred_name'].values)
    assert all(rc_meta_edge_df['succ_name'].values == meta_edge_df['succ_name'].values)

    # 验证_gen_dgl_graph的正确性
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

    rc_edge_df = rc_meta_edge_df.iloc[dgl_graph.edata['meta_edge_id']].reset_index()
    edge_df = graph.edge_df()
    columns = edge_df.columns.intersection(rc_edge_df.columns)
    print(columns)
    assert all(rc_edge_df[columns] == edge_df[columns])


if __name__ == '__main__':
    test()
