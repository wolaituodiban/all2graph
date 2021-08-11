from typing import Dict, List, Tuple, Union


class JsonGraph:
    def __init__(
            self,
            names: List[str] = None,
            values: List[Union[Dict, List, str, int, float, None]] = None,
            edges: List[Tuple[int, int]] = None
    ):
        self.names = list(names or [])
        self.values = list(values or [])
        self.edges = list(edges or [])

    @property
    def num_nodes(self):
        return len(self.names)

    @property
    def num_edges(self):
        return len(self.edges)

    def insert_node(
            self,
            name: str,
            value: Union[Dict, List, str, int, float, None],
            preds: List[int] = None,
            succs: List[int] = None
    ) -> int:
        node_id = len(self.names)
        self.names.append(name)
        self.values.append(value)
        if preds is not None:
            for pred in preds:
                self.edges.append((pred, node_id))
        if succs is not None:
            for succ in succs:
                self.edges.append((succ, node_id))
        return node_id

    def insert_json(
            self,
            name: str,
            value: Union[Dict, List, str, int, float, None],
            preds: List[int] = None,
            succs: List[int] = None
    ):
        node_id = self.insert_node(name, value, preds, succs)
        if isinstance(value, dict):
            for k, v in value.items():
                self.insert_json(k, v, preds=[node_id])
        elif isinstance(value, list):
            for v in value:
                self.insert_json(name, v, preds=[node_id])
