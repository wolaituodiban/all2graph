import string
from typing import List, Iterable
import re

from ..globals import SEP


class Tokenizer:
    def __init__(self, camel_to_snake, join_token):
        if camel_to_snake:
            self.camel_to_snake_pattern = re.compile(r'([a-z])([A-Z])')
        else:
            self.camel_to_snake_pattern = None
        self.join_token = join_token

    def camel_to_snake(self, s, lower=True):
        snake = self.camel_to_snake_pattern.sub(r'\1_\2', s)
        if lower:
            snake = snake.lower()
        return snake

    def cut(self, s: str, **kwargs) -> Iterable[str]:
        return []

    def lcut(self, s: str, **kwargs) -> List[str]:
        return []

    def join(self, x: List[str], **kwargs) -> str:
        return self.join_token.join(x)


null_tokenizer = Tokenizer(camel_to_snake=False, join_token=SEP)

try:
    import jieba

    class JiebaTokenizer(Tokenizer):
        def __init__(self, camel_to_snake=False, dictionary=None, stopwords=None, join_token=''):
            super().__init__(camel_to_snake=camel_to_snake, join_token=join_token)
            if dictionary is not None:
                tokenizer = jieba.Tokenizer(dictionary)
            else:
                tokenizer = jieba.Tokenizer()
            tokenizer.initialize()
            tokenizer.lock = None
            self.tokenizer = tokenizer
            self.stopwords = stopwords

        def cut(self, s: str, **kwargs) -> Iterable[str]:
            if self.camel_to_snake_pattern is not None:
                s = self.camel_to_snake(s)
            output = self.tokenizer.cut(s, **kwargs)
            if self.stopwords is not None:
                output = (x for x in output if x not in self.stopwords)
            return output

        def lcut(self, s: str, **kwargs) -> List[str]:
            return list(self.cut(s, **kwargs))

        def __repr__(self):
            return '{}(kill_camel={})'.format(self.__class__.__name__, self.camel_to_snake_pattern is not None)

    default_tokenizer = JiebaTokenizer(camel_to_snake=True, stopwords={SEP}.union(string.punctuation), join_token=SEP)
except ImportError:
    default_tokenizer = None
