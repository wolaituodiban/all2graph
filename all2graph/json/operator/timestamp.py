import sys
from datetime import datetime
from .operator import Operator


class Timestamp(Operator):
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
        return '{}(name={}, units={})'.format(
            self.__class__.__name__, self.name, list(self.units))


class Timestamp2(Operator):
    def __init__(self, _format, units):
        super().__init__()
        self.format = _format
        self.units = units

    def __call__(self, obj, now: datetime = None, **kwargs):
        output = {}
        try:
            time = datetime.strptime(obj, self.format)
        except ValueError:
            print('{}不满足{}格式'.format(obj, self.format), file=sys.stderr)
            return None
        if now is not None:
            diff = now - time
            output['diff_day'] = diff.days + int(diff.seconds > 0)
        for unit, feat_name in self.units.items():
            if unit == 'weekday':
                output[feat_name] = time.weekday()
            elif hasattr(time, unit):
                output[feat_name] = getattr(time, unit)
        return output

    def __repr__(self):
        return '{}(units={})'.format(self.__class__.__name__, list(self.units))


class Timestamp3(Operator):
    def __init__(self, name, _format, units, error=False):
        super().__init__()
        self.name = name
        self.format = _format
        self.units = list(units)
        self.error = error

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
                obj['diff_day'.format(self.name)] = diff.days + int(diff.seconds > 0)
            for unit in self.units:
                if unit == 'weekday':
                    obj[unit] = time.weekday()
                elif hasattr(time, unit):
                    obj[unit] = getattr(time, unit)
        elif self.error:
            raise KeyError('object do not have ({}): {}'.format(self.name, obj))
        return obj

    def __repr__(self):
        return '{}(name={}, units={}, error={})'.format(
            self.__class__.__name__, self.name, list(self.units), self.error)
