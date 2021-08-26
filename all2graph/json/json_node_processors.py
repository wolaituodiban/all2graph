from datetime import datetime

from ..version import __version__


class JsonNodeProcessor:
    def __init__(self):
        self.version = __version__

    def __call__(self, obj, **kwargs):
        raise NotImplementedError

    def __repr__(self):
        return '{}()'.format(self.__class__.__name__)


class TimeProcessor(JsonNodeProcessor):
    def __init__(self, name, _format, units):
        super().__init__()
        self.name = name
        self.format = _format
        self.units = {
            unit: '{}_{}'.format(self.name, unit) for unit in units
        }

    def __call__(self, obj, sample_time: datetime = None, **kwargs):
        if self.name in obj:
            date_string = obj[self.name]
            try:
                time = datetime.strptime(date_string, self.format)
            except ValueError:
                # todo 写个warning
                return None
            if sample_time is not None:
                diff = sample_time - time
                obj['{}_diff_day'.format(self.name)] = diff.days + int(diff.seconds > 0)
            for unit, feat_name in self.units.items():
                if unit == 'weekday':
                    obj[feat_name] = time.weekday()
                elif hasattr(time, unit):
                    obj[feat_name] = getattr(time, unit)
        return obj

    def __repr__(self):
        return '{}(name={}, units={})'.format(self.__class__.__name__, self.name, list(self.units))


class Delete(JsonNodeProcessor):
    def __init__(self, names):
        super().__init__()
        self.names = names

    def __call__(self, obj, **kwargs):
        for name in self.names:
            if name in obj:
                del obj[name]
        return obj

    def __repr__(self):
        return '{}(names={})'.format(self.__class__.__name__, self.names)


class Sorted(JsonNodeProcessor):
    def __init__(self, key=None, reverse=False):
        super().__init__()
        self.key = key
        self.reverse = reverse

    def __call__(self, obj, **kwargs):
        if self.key is None:
            return sorted(obj, key=lambda x: x[self.key], reverse=self.reverse)
        else:
            return sorted(obj, reverse=self.reverse)


class Split(JsonNodeProcessor):
    def __init__(self, sep, maxsplit=-1):
        super().__init__()
        self.sep = sep
        self.maxsplit = maxsplit

    def __call__(self, obj: str, **kwargs):
        if isinstance(obj, str):
            return obj.split(sep=self.sep, maxsplit=self.maxsplit)
        else:
            return obj

    def __repr__(self):
        return "{}(sep='{}', maxsplit={})".format(self.__class__.__name__, self.sep, self.maxsplit)


class Lower(JsonNodeProcessor):
    def __init__(self):
        super().__init__()

    def __call__(self, obj, **kwargs):
        if isinstance(obj, str):
            return obj.lower()
        else:
            return obj
