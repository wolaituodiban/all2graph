import string
from typing import List, Iterable

from ..globals import SEP
from jsonpromax import camel_to_snake


class Tokenizer:
    def __init__(self, _camel_to_snake: bool):
        self.camel_to_snake_pattern = _camel_to_snake

    @staticmethod
    def camel_to_snake(*args, **kwargs):
        return camel_to_snake(*args, **kwargs)

    def cut(self, s: str, **kwargs) -> Iterable[str]:
        return []

    def lcut(self, s: str, **kwargs) -> List[str]:
        return []


null_tokenizer = Tokenizer(_camel_to_snake=False)

try:
    import jieba

    class JiebaTokenizer(Tokenizer):
        def __init__(self, _camel_to_snake=False, dictionary=None, stopwords=None):
            super().__init__(_camel_to_snake=_camel_to_snake)
            if dictionary is not None:
                tokenizer = jieba.Tokenizer(dictionary)
            else:
                tokenizer = jieba.Tokenizer()
            tokenizer.initialize()
            tokenizer.lock = None
            self.tokenizer = tokenizer
            self.stopwords = stopwords

        def cut(self, s: str, **kwargs) -> Iterable[str]:
            if self.camel_to_snake_pattern:
                s = self.camel_to_snake(s)
            output = self.tokenizer.cut(s, **kwargs)
            if self.stopwords is not None:
                output = (x for x in output if x not in self.stopwords)
            return output

        def lcut(self, s: str, **kwargs) -> List[str]:
            return list(self.cut(s, **kwargs))

        def __repr__(self):
            return '{}(kill_camel={})'.format(self.__class__.__name__, self.camel_to_snake_pattern)
except ImportError:
    JiebaTokenizer = None
_default_tokenizer = None


def default_tokenizer():
    global _default_tokenizer
    if _default_tokenizer is None and JiebaTokenizer is not None:
        stopwords = set(string.ascii_letters).union(string.punctuation).union(string.digits).union([' ', '__', SEP])
        stopwords = stopwords.union("，。/；‘【】、·-=～——+）（*&……%¥#@！「」｜“：？》《")
        _default_tokenizer = JiebaTokenizer(_camel_to_snake=True, stopwords=stopwords)
    return _default_tokenizer
