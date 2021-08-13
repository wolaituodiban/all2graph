import json
from typing import Dict, Tuple
import numpy as np
import pandas as pd
from toad.utils.progress import Progress
from .meta_graph import MetaGraph
from ..meta_node import Index, JsonValue
from ..meta_edge import MetaEdge
from ..stats import ECDF
from ..graph import Graph


class JsonGraph(MetaGraph):
    INDEX_NODES = 'index_nodes'
    """解析json，并生成表述json结构的元图"""
    def __init__(self, nodes: Dict[str, JsonValue], edges: Dict[Tuple[str, str], MetaEdge],
                 index_nodes: Dict[str, Index] = None, **kwargs):
        assert all(isinstance(n, (Index, JsonValue)) for n in nodes.values())
        if index_nodes is not None and len(index_nodes) > 0:
            assert all(isinstance(n, (Index, Index)) for n in index_nodes.values())
            assert list({n.num_samples for n in nodes.values()}) == list({n.num_samples for n in index_nodes.values()})
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
            obj[cls.INDEX_NODES] = {k: Index.from_json(v) for k, v in obj[cls.INDEX_NODES].items()}
        return super().from_json(obj)

    @classmethod
    def from_data(cls, sample_times, jsons, index_names=None, **kwargs):
        """

        :params sample_times:
        :params graph:
        """
        assert len(sample_times) == len(jsons)
        json_graph = Graph()
        for i, value in enumerate(Progress(jsons.values)):
            if isinstance(value, str):
                value = json.loads(value)
            json_graph.insert_patch(i, 'graph', value)

        # 创建nodes
        node_df = pd.DataFrame(
            {
                'sample_id': json_graph.patch_ids,
                'name': json_graph.names,
                'value': json_graph.values,
                'sample_time': sample_times.iloc[json_graph.patch_ids]
            }
        )
        nodes = {}
        index_nodes = {}
        for name, group_df in Progress(node_df.groupby('name')):
            if index_names is not None and name in index_names:
                index_nodes[name] = Index.from_data(
                    num_samples=len(jsons), sample_ids=group_df.sample_id, values=group_df.value,
                    sample_times=group_df.sample_time
                )
            else:
                nodes[name] = JsonValue.from_data(
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
    def reduce(cls, graphs, **kwargs):
        num_samples = 0
        nodes = {}
        edges = {}
        index_nodes = {}
        for graph in graphs:
            num_samples += graph.num_samples
            for k, v in graph.nodes.items():
                if isinstance(v, JsonValue):
                    if k in nodes:
                        nodes[k].append(v)
                    else:
                        nodes[k] = [v]
                else:
                    assert isinstance(v, Index)
                    if k in index_nodes:
                        index_nodes[k].append(v)
                    else:
                        index_nodes[k] = [v]

            for k, v in graph.edges.items():
                if k in nodes:
                    edges[k].append(v)
                else:
                    edges[k] = [v]

        nodes = {k: JsonValue.reduce(v) for k, v in Progress(nodes.items())}
        index_nodes = {k: Index.reduce(v) for k, v in Progress(index_nodes.items())}
        edges = {k: MetaEdge.reduce(v) for k, v in Progress(edges.items())}

        for k in Progress(nodes):
            if nodes[k].num_samples < num_samples:
                nodes[k].node_freq = ECDF.reduce(
                    [
                        nodes[k].node_freq,
                        ECDF.from_data(np.zeros(num_samples-nodes[k].num_samples), **kwargs)]
                )

        for k in Progress(index_nodes):
            if index_nodes[k].num_samples < num_samples:
                index_nodes[k].node_freq = ECDF.reduce(
                    [
                        index_nodes[k].node_freq,
                        ECDF.from_data(np.zeros(num_samples-index_nodes[k].num_samples), **kwargs)]
                )

        for k in Progress(edges):
            if edges[k].num_samples < num_samples:
                edges[k].freq = ECDF.reduce(
                    [
                        edges[k].freq,
                        ECDF.from_data(np.zeros(num_samples-edges[k].num_samples), **kwargs)]
                )
        return super().reduce(graphs, nodes=nodes, edges=edges, index_nodes=index_nodes, **kwargs)
