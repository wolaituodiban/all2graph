import sys
import traceback
from datetime import datetime
from .operator import Operator


class Timestamp(Operator):
    def __init__(self, name, _format, units, second=False, error=True, warning=True):
        super().__init__()
        self.name = name
        self.format = _format
        self.units = {
            unit: '{}_{}'.format(self.name, unit) for unit in units
        }
        self.second = second
        self.error = error
        self.warning = warning

    def __call__(self, obj, now: datetime = None, **kwargs):
        try:
            date_string = obj[self.name]
        except (ValueError, IndexError, TypeError, KeyError) as e:
            if self.error:
                raise e
            elif self.warning:
                traceback.print_exc(file=sys.stderr)
            return obj
        try:
            time = datetime.strptime(date_string, self.format)
        except (ValueError, IndexError, TypeError, KeyError) as e:
            if self.error:
                raise e
            elif self.warning:
                traceback.print_exc(file=sys.stderr)
            return obj
        if now is not None:
            diff = now - time
            if self.second:
                obj['diff_second'] = diff.total_seconds()
            else:
                obj['diff_day'] = diff.days + int(diff.seconds > 0)
        for unit, feat_name in self.units.items():
            if unit == 'weekday':
                obj[feat_name] = time.weekday()
            elif hasattr(time, unit):
                obj[feat_name] = getattr(time, unit)
        return obj

    def __repr__(self):
        return '{}(name={}, units={}, diff_units={})'.format(
            self.__class__.__name__, self.name, list(self.units), 'second' if self.second else 'day')


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
    def __init__(self, name, _format, units, second=True, error=True, warning=True):
        super().__init__()
        self.name = name
        self.format = _format
        self.units = list(units)
        self.second = second
        self.error = error
        self.warning = warning

    def __call__(self, obj, now: datetime = None, **kwargs):
        try:
            date_string = obj[self.name]
        except (ValueError, IndexError, TypeError, KeyError)as e:
            if self.error:
                raise e
            elif self.warning:
                traceback.print_exc(file=sys.stderr)
            return obj
        try:
            time = datetime.strptime(date_string, self.format)
        except (ValueError, IndexError, TypeError, KeyError) as e:
            if self.error:
                raise e
            elif self.warning:
                traceback.print_exc(file=sys.stderr)
            return None
        if now is not None:
            diff = now - time
            if self.second:
                obj['diff_second'] = diff.total_seconds()
            else:
                obj['diff_day'] = diff.days + int(diff.seconds > 0)
        for unit in self.units:
            if unit == 'weekday':
                obj[unit] = time.weekday()
            elif hasattr(time, unit):
                obj[unit] = getattr(time, unit)
        return obj

    def __repr__(self):
        return '{}(name={}, units={}, diff_units={})'.format(
            self.__class__.__name__, self.name, list(self.units), 'second' if self.second else 'day')
