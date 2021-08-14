from typing import Dict, List, Union


class CallBack:
    def __call__(
            self,
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
