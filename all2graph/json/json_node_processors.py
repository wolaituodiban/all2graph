from datetime import datetime

from ..version import __version__


class JsonNodeProcessor:
    def __init__(self):
        self.version = __version__

    def __call__(self, obj, **kwargs):
        raise NotImplementedError

    def __repr__(self):
        return self.__class__.__name__


class TimeProcessor(JsonNodeProcessor):
    def __init__(self, name, _format, units):
        super().__init__()
        self.name = name
        self.format = _format
        self.units = units

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
            for unit in self.units:
                if unit == 'weekday':
                    obj['{}_{}'.format(self.name, unit)] = time.weekday()
                elif hasattr(time, unit):
                    obj['{}_{}'.format(self.name, unit)] = getattr(time, unit)
        return obj


class Delete(JsonNodeProcessor):
    def __init__(self, names):
        super().__init__()
        self.names = names

    def __call__(self, obj, **kwargs):
        for name in self.names:
            if name in obj:
                del obj[name]
        return obj


class GetAttr(JsonNodeProcessor):
    def __init__(self, name):
        super().__init__()
        self.name = name

    def __call__(self, obj, **kwargs):
        if self.name in obj:
            return obj[self.name]
        else:
            return None


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


class SplitString(JsonNodeProcessor):
    def __init__(self, sep, maxsplit=-1):
        super().__init__()
        self.sep = sep
        self.maxsplit = maxsplit

    def __call__(self, obj: str, **kwargs):
        if isinstance(obj, str):
            return obj.split(sep=self.sep, maxsplit=self.maxsplit)
        else:
            return obj


class Lower(JsonNodeProcessor):
    def __init__(self):
        super().__init__()

    def __call__(self, obj, **kwargs):
        if isinstance(obj, str):
            return obj.lower()
        else:
            return obj
