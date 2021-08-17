import json
from typing import Dict, Tuple
import numpy as np
import pandas as pd
from toad.utils.progress import Progress
from .meta_graph import MetaGraph
from .meta_node import MetaIndex, MetaJsonValue
from .meta_edge import MetaEdge
from ..stats import ECDF
from ..graph import JsonGraph


class MetaJsonGraph(MetaGraph):
    INDEX_NODES = 'index_nodes'
    """解析json，并生成表述json结构的元图"""
    def __init__(self, nodes: Dict[str, MetaJsonValue], edges: Dict[Tuple[str, str], MetaEdge],
                 index_nodes: Dict[str, MetaIndex] = None, **kwargs):
        assert all(isinstance(n, (MetaIndex, MetaJsonValue)) for n in nodes.values())
        if index_nodes is not None and len(index_nodes) > 0:
            assert all(isinstance(n, MetaIndex) for n in index_nodes.values())
        else:
            index_nodes = None
        super().__init__(nodes=nodes, edges=edges, **kwargs)
        self.index_nodes = index_nodes

    def __eq__(self, other):
        return super().__eq__(other) and self.index_nodes == other.index_nodes

    def to_json(self) -> dict:
        output = super().to_json()
        if self.index_nodes is not None:
            output.update({
                self.INDEX_NODES: {k: v.to_json() for k, v in self.index_nodes.items()},
            })
        return output

    @classmethod
    def from_json(cls, obj, **kwargs):
        """

        :param obj: json对象
        :return:
        """
        if isinstance(obj, str):
            obj = json.loads(obj)
        else:
            obj = dict(obj)

        if cls.INDEX_NODES in obj:
            obj[cls.INDEX_NODES] = {k: MetaIndex.from_json(v) for k, v in obj[cls.INDEX_NODES].items()}
        return super().from_json(obj)

    @classmethod
    def from_data(cls, sample_times, jsons, index_names=None, **kwargs):
        """

        :params sample_times:
        :params graph:
        """
        assert len(sample_times) == len(jsons)
        json_graph = JsonGraph()
        for i, value in enumerate(Progress(jsons.values)):
            if isinstance(value, str):
                value = json.loads(value)
            json_graph.insert_component(i, 'graph', value)

        # 创建nodes
        node_df = pd.DataFrame(
            {
                'sample_id': json_graph.component_ids,
                'name': json_graph.names,
                'value': json_graph.values,
                'sample_time': sample_times.iloc[json_graph.component_ids]
            }
        )
        nodes = {}
        index_nodes = {}
        for name, group_df in Progress(node_df.groupby('name')):
            if index_names is not None and name in index_names:
                index_nodes[name] = MetaIndex.from_data(
                    num_samples=len(jsons), sample_ids=group_df.sample_id, values=group_df.value,
                    sample_times=group_df.sample_time
                )
            else:
                nodes[name] = MetaJsonValue.from_data(
                    num_samples=len(jsons), sample_ids=group_df.sample_id, values=group_df.value,
                    sample_times=group_df.sample_time
                )

        # 创建edges
        edge_df = pd.DataFrame(
            {
                'sample_id': node_df.sample_id.iloc[json_graph.preds],
                'pred_name': node_df.name.iloc[json_graph.preds],
                'succ_name': node_df.name.iloc[json_graph.succs]
            }
        )
        edges = {}
        for (pred, succ), group_df in Progress(edge_df.groupby(['pred_name', 'succ_name'])):
            edges[(pred, succ)] = MetaEdge.from_data(len(jsons), group_df.sample_id, **kwargs)

        return super().from_data(nodes=nodes, edges=edges, index_nodes=index_nodes, **kwargs)

    @classmethod
    def reduce(cls, graphs, weights=None, **kwargs):
        if weights is None:
            weights = np.full(len(graphs), 1 / len(graphs))
        else:
            weights = np.array(weights) / sum(weights)

        temp_nodes = {}
        temp_edges = {}
        temp_indices = {}
        for w, graph in zip(weights, graphs):
            for k, v in graph.nodes.items():
                if isinstance(v, MetaJsonValue):
                    if k in temp_nodes:
                        temp_nodes[k][0].append(w)
                        temp_nodes[k][1].append(v)
                    else:
                        temp_nodes[k] = ([w], [v])
                else:
                    assert isinstance(v, MetaIndex)
                    if k in temp_indices:
                        temp_indices[k][0].append(w)
                        temp_indices[k][1].append(v)
                    else:
                        temp_indices[k] = ([w], [v])

            for k, v in graph.edges.items():
                if k in temp_nodes:
                    temp_edges[k][0].append(w)
                    temp_edges[k][1].append(v)
                else:
                    temp_edges[k] = ([w], [v])

        nodes = {k: MetaJsonValue.reduce(v, weights=w, **kwargs) for k, (w, v) in Progress(temp_nodes.items())}
        indices = {k: MetaIndex.reduce(v, weights=w, **kwargs) for k, (w, v) in Progress(temp_indices.items())}
        edges = {k: MetaEdge.reduce(v, weights=w, **kwargs) for k, (w, v) in Progress(temp_edges.items())}

        for k in temp_nodes:
            w_sum = sum(temp_nodes[k][0])
            if w_sum < 1:
                nodes[k].freq = ECDF.reduce(
                    [
                        nodes[k].freq,
                        ECDF([0], [1], initialized=True)
                    ],
                    weights=[w_sum, 1 - w_sum],
                    **kwargs
                )

        for k in temp_indices:
            w_sum = sum(temp_indices[k][0])
            if w_sum < 1:
                indices[k].freq = ECDF.reduce(
                    [
                        indices[k].freq,
                        ECDF([0], [1], initialized=True)
                    ],
                    weights=[w_sum, 1 - w_sum],
                    **kwargs
                )

        for k in temp_edges:
            w_sum = sum(temp_edges[k][0])
            if w_sum < 1:
                edges[k].freq = ECDF.reduce(
                    [
                        edges[k].freq,
                        ECDF([0], [1], initialized=True)
                    ],
                    weights=[w_sum, 1 - w_sum],
                    **kwargs
                )
        return super().reduce(graphs, nodes=nodes, edges=edges, index_nodes=indices, **kwargs)
