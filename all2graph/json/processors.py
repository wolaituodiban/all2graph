import json
from datetime import datetime
from typing import Iterable, Dict, Union, List

import pandas as pd


class Processor:
    def process(self, *args, **kwargs):
        raise NotImplementedError


class TimeProcessor(Processor):
    def __init__(self, format, units):
        self.format = format
        self.units = units

    def process(self, date_string: str, sample_time: datetime):
        try:
            time = datetime.strptime(date_string, self.format)
        except:
            # todo 写个warning
            return None
        output = {
            'diff_days': (sample_time - time).days
        }
        for unit in self.units:
            if hasattr(time, unit):
                output[unit] = getattr(time, unit)
        return output


def parsing_json_path(path: str, obj: Union[dict, list]):
    star_pos = path.find('*')
    if star_pos < 0:
        return [path]
    else:
        output = []
        root_path = path[:star_pos-1]
        array = eval(root_path.replace('$', 'obj'))
        for i, item in enumerate(array):
            output += [
                p.replace('$', root_path)
                for p in parsing_json_path('$' + path[star_pos-1:].replace('*', str(i), 1), item)
            ]
        return output


class ProcessorTree:
    def __init__(self, name):
        self.name = name
        self.processors: List[Processor] = []
        self.childs: List[ProcessorTree] = []

    def insert(self, json_path: str, processor: Processor):
        pass










class JsonPreProcessor:
    def __init__(self, root_name, sample_time_col=None, sample_time_format=None,
                 time_processors: Dict[str, TimeProcessor] = None, processors: Dict[str, Processor] = None):
        self.root_name = root_name
        self.sample_time_col = sample_time_col
        self.sample_time_format = sample_time_format
        self.time_paths = dict(time_processors or {})

    def __call__(self, df: pd.DataFrame) -> Iterable:
        if self.sample_time_col is None:
            for value in df[self.root_name]:
                yield json.loads(value)
        else:
            df = df[[self.root_name, self.sample_time_col]].copy()
            for row in df.itertuples():
                obj = json.loads(row[1])
                sample_time = datetime.strptime(row[2], self.sample_time_format)
                for path, pros in self.time_paths.items():
                    field = eval('obj'+path)
                    field = pros.process(field, sample_time)
                    exec("obj{} = field".format(path))
                yield obj

