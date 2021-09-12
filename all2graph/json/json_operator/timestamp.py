import sys
from datetime import datetime
from .json_operator import JsonOperator


class Timestamp(JsonOperator):
    def __init__(self, name, _format, units):
        super().__init__()
        self.name = name
        self.format = _format
        self.units = {
            unit: '{}_{}'.format(self.name, unit) for unit in units
        }

    def __call__(self, obj, now: datetime = None, **kwargs):
        if self.name in obj:
            date_string = obj[self.name]
            try:
                time = datetime.strptime(date_string, self.format)
            except ValueError:
                print('{}不满足{}格式'.format(date_string, self.format), file=sys.stderr)
                return None
            if now is not None:
                diff = now - time
                obj['{}_diff_day'.format(self.name)] = diff.days + int(diff.seconds > 0)
            for unit, feat_name in self.units.items():
                if unit == 'weekday':
                    obj[feat_name] = time.weekday()
                elif hasattr(time, unit):
                    obj[feat_name] = getattr(time, unit)
        return obj

    def __repr__(self):
        return '{}(name={}, units={})'.format(self.__class__.__name__, self.name, list(self.units))
