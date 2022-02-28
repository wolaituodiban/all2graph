from typing import Dict
from abc import abstractproperty, abstractmethod
from ..meta_struct import MetaStruct
from ..stats import ECDF


class MetaInfo(MetaStruct):
    @abstractmethod
    def dictionary(self, **kwargs) -> Dict[str, int]:
        raise NotImplementedError

    @abstractproperty
    def numbers(self) -> Dict[str, ECDF]:
        raise NotImplementedError
