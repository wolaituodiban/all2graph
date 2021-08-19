from ..meta_struct import MetaStruct


class Distribution(MetaStruct):
    @classmethod
    def from_json(cls, obj):
        return super().from_json(obj)
