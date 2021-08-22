import json
from typing import Iterable

import pandas as pd


class JsonPreProcessor:
    def __init__(self, root_name):
        self.root_name = root_name

    def __call__(self, df: pd.DataFrame) -> Iterable:
        return map(json.loads, df[self.root_name])
