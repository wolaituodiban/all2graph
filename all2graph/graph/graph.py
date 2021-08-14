from typing import Dict, List, Union, Tuple


def default_callback(
        node_id: int,
        patch_id: int,
        name: str,
        value: Union[Dict, List, str, int, float, bool, None],
        preds: Union[List[int], None],
        succs: Union[List[int], None],
):
    if isinstance(value, dict):
        for k, v in value.items():
            yield patch_id, k, v, [node_id], None
    elif isinstance(value, list):
        for v in value:
            yield patch_id, name, v, [node_id], None


class Graph:
    def __init__(self):
        self.patch_ids: List[int] = []
        self.names: List[str] = []
        self.values: List[Union[Dict, List, str, int, float, None]] = []
        self.preds: List[int] = []
        self.succs: List[int] = []

    @property
    def num_nodes(self):
        return len(self.names)

    @property
    def num_edges(self):
        return len(self.preds)

    def insert_node(
            self,
            patch_id: int,
            name: str,
            value: Union[Dict, List, str, int, float, None],
            preds: List[int] = None,
            succs: List[int] = None
    ) -> int:
        node_id = len(self.names)
        self.patch_ids.append(patch_id)
        self.names.append(name)
        self.values.append(value)
        if preds is not None:
            for pred in preds:
                self.preds.append(pred)
                self.succs.append(node_id)
        if succs is not None:
            for succ in succs:
                self.preds.append(node_id)
                self.succs.append(succ)
        return node_id

    def insert_patch(
            self,
            patch_id: int,
            name: str,
            value: Union[Dict, List, str, int, float, None],
            preds: List[int] = None,
            succs: List[int] = None,
            callback=default_callback
    ):
        node_id = self.insert_node(patch_id, name, value, preds, succs)
        for args in callback(
                node_id=node_id, patch_id=patch_id, name=name, value=value, preds=preds, succs=succs
        ):
            self.insert_patch(*args, callback=callback)
