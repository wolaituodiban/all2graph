import os
import json
import pandas as pd
import torch
from all2graph import MetaGraph, EPSILON, NULL, META
from all2graph.json import JsonResolver
from all2graph.transformer import Transformer
from all2graph.utils import Timer
import json_tools


path = os.path.dirname(__file__)
path = os.path.dirname(path)
path = os.path.dirname(path)

meta_graph_path = os.path.join(path, 'test_data', 'meta_graph.json')
with open(meta_graph_path, 'r') as file:
    meta_graph = MetaGraph.from_json(json.load(file))


def test_init():
    Transformer.from_meta_graph(
        meta_graph, min_df=0.01, max_df=0.95, top_k=100, top_method='max_tfidf', segment_name=False
    )


csv_path = os.path.join(path, 'test_data', 'MensShoePrices.csv')
df = pd.read_csv(csv_path, nrows=64)

resolver = JsonResolver(
    root_name='json', flatten_dict=True, local_index_names={'name'}, segmentation=True, self_loop=True,
    list_inner_degree=1
)
graph, global_index_mapper, local_index_mappers = resolver.resolve(
    list(map(json.loads, df.json)), progress_bar=True
)


def test_non_segment():
    trans = Transformer.from_meta_graph(
        meta_graph, segment_name=False
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
    assert (rc_meta_edge_df['component_id'].values
            == rc_meta_node_df['component_id'].iloc[rc_meta_edge_df['succ_meta_node_id']].values).all()
    rc_meta_edge_df['pred_name'] = rc_meta_node_df['name'].iloc[rc_meta_edge_df['pred_meta_node_id']].values
    rc_meta_edge_df['succ_name'] = rc_meta_node_df['name'].iloc[rc_meta_edge_df['succ_meta_node_id']].values

    meta_edge_df = graph.meta_edge_df()
    assert (rc_meta_edge_df['pred_name'].values == meta_edge_df['pred_name'].str.lower().values).all(), json_tools.diff(
        rc_meta_edge_df['pred_name'].values.tolist(), meta_edge_df['pred_name'].values.tolist()
    )
    assert (rc_meta_edge_df['succ_name'].values == meta_edge_df['succ_name'].str.lower().values).all()

    # 验证_gen_dgl_graph的正确性
    rc_graph = trans.graph_from_dgl(dgl_meta_graph, dgl_graph)
    assert graph.component_ids == rc_graph.component_ids
    assert graph.preds == rc_graph.preds
    assert graph.succs == rc_graph.succs
    assert rc_graph.names == [item.lower() for item in graph.names]
    values = pd.Series(graph.values)
    rc_values = pd.Series(rc_graph.values)
    numbers = pd.to_numeric(values, errors='coerce')
    str_mask = values.apply(lambda x: isinstance(x, str) and x in trans.string_mapper) & numbers.isna()
    assert (values[str_mask] == rc_values[str_mask]).all()

    rc_edge_df = rc_meta_edge_df.iloc[dgl_graph.edata['meta_edge_id']].reset_index()
    edge_df = graph.edge_df()
    assert (edge_df['component_id'] == rc_edge_df['component_id']).all()
    assert (edge_df['pred_name'].str.lower() == rc_edge_df['pred_name']).all()
    assert (edge_df['succ_name'].str.lower() == rc_edge_df['succ_name']).all()


def test_segment():
    trans1 = Transformer.from_meta_graph(meta_graph, segment_name=False)
    with Timer('non_segment'):
        dgl_meta_graph1, dgl_graph1 = trans1.graph_to_dgl(graph)

    trans2 = Transformer.from_meta_graph(meta_graph, segment_name=True)
    with Timer('segment'):
        dgl_meta_graph2, dgl_graph2 = trans2.graph_to_dgl(graph)

    assert (dgl_graph1.ndata['meta_node_id'] == dgl_graph2.ndata['meta_node_id']).all()
    assert (dgl_graph1.ndata['value'] == dgl_graph2.ndata['value']).all()
    assert torch.max(torch.abs(dgl_graph1.ndata['value'] - dgl_graph2.ndata['value'])) <= EPSILON
    assert dgl_graph1.edges[0] == dgl_graph2.edges[0]
    assert dgl_graph1.edges[1] == dgl_graph2.edges[1]

    assert (dgl_meta_graph1.ndata['component_id']
            == dgl_meta_graph2.ndata['component_id'][:dgl_meta_graph1.num_nodes()]).all()
    assert (dgl_meta_graph1.ndata['component_id']
            == dgl_meta_graph2.ndata['component_id'][:dgl_meta_graph1.num_nodes()]).all()
    assert (dgl_meta_graph2.ndata['name'] != trans2.encode(NULL)).all()

    rc_graph = trans2.graph_from_dgl(meta_graph=dgl_meta_graph2, graph=dgl_graph2)
    assert rc_graph.names == [x.lower() for x in graph.names]


if __name__ == '__main__':
    test_init()
    test_non_segment()
    test_segment()
