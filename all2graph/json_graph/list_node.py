from ..meta_graph import MetaNode


class ListNode(MetaNode):
    def to_json(self) -> dict:
        return super().to_json()

    @classmethod
    def from_json(cls, obj):
        return super().from_json(obj)

    @classmethod
    def from_data(cls, **kwargs):
        raise NotImplementedError
