from typing import List, Dict
from abc import abstractproperty
from ..meta_struct import MetaStruct
from ..stats import ECDF


class MetaInfo(MetaStruct):
    @abstractproperty
    def dictionary(self) -> List[str]:
        raise NotImplementedError

    @abstractproperty
    def values(self) -> Dict[str, ECDF]:
        raise NotImplementedError
