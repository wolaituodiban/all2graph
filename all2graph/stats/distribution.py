from abc import abstractmethod
from ..meta_struct import MetaStruct


class Distribution(MetaStruct):
    def __init__(self, num_samples, **kwargs):
        super().__init__(**kwargs)
        self.num_samples = num_samples

    def __eq__(self, other):
        return super().__eq__(other) and self.num_samples == other.num_samples

    @abstractmethod
    def to_json(self) -> dict:
        output = super().to_json()
        output['num_samples'] = self.num_samples
        return output
