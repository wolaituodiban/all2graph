import json
from typing import Dict, Tuple
import pandas as pd
from toad.utils.progress import Progress
from .meta_graph import MetaGraph
from ..meta_node import Index, JsonValue
from ..meta_edge import MetaEdge
from ..stats import ECDF
from ..graph import Graph


class JsonGraph(MetaGraph):
    """解析json，并生成表述json结构的元图"""
    def __init__(self, nodes: Dict[str, JsonValue], edges: Dict[Tuple[str, str], MetaEdge], **kwargs):
        assert all(isinstance(n, (Index, JsonValue)) for n in nodes.values())
        super().__init__(nodes=nodes, edges=edges, **kwargs)

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
        for name, group_df in Progress(node_df.groupby('name')):
            if index_names is not None and name in index_names:
                node_cls = Index
            else:
                node_cls = JsonValue
            nodes[name] = node_cls.from_data(
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

        return super().from_data(nodes=nodes, edges=edges, **kwargs)

    @classmethod
    def reduce(cls, graphs, **kwargs):
        num_samples = 0
        nodes = {}
        edges = {}
        for graph in graphs:

            for k, v in graph.nodes.items():
                if k in nodes:
                    nodes[k].append(v)
                else:
                    nodes[k] = [v]

            for k, v in graph.edges.items():
                if k in nodes:
                    edges[k].append(v)
                else:
                    edges[k] = [v]

        nodes = {k: JsonValue.reduce(v) for k, v in nodes.items()}
        edges = {k: MetaEdge.reduce(v) for k, v in edges.items()}

        for k in nodes:
            if nodes[k].num_samples < num_samples:
                nodes[k].node_freq = ECDF.reduce(
                    [
                        nodes[k].node_freq,
                        ECDF.from_data([0] * (num_samples-nodes[k].node_freq.num_samples), **kwargs)]
                )

        for k in edges:
            if edges[k].num_samples < num_samples:
                edges[k].freq = ECDF.reduce(
                    [
                        edges[k].freq,
                        ECDF.from_data([0] * (num_samples-edges[k].node_freq.num_samples), **kwargs)]
                )
        return super().reduce(graphs, nodes=nodes, edges=edges, **kwargs)
