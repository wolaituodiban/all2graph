from abc import abstractmethod
from typing import List, Iterable
import re

from ..globals import SEP


class Tokenizer:
    def __init__(self, kill_camel, join_token):
        if kill_camel:
            self.kill_camel_pattern = re.compile(r'([a-z])([A-Z])')
        else:
            self.kill_camel_pattern = None
        self.join_token = join_token

    def kill_camel(self, s):
        return self.kill_camel_pattern.sub(r'\1 \2', s)

    def cut(self, s: str, **kwargs) -> Iterable[str]:
        raise NotImplementedError

    def lcut(self, s: str, **kwargs) -> List[str]:
        raise NotImplementedError

    def join(self, x: List[str], **kwargs) -> str:
        return self.join_token.join(x)


try:
    import jieba

    class JiebaTokenizer(Tokenizer):
        def __init__(self, kill_camel=False, dictionary=None, stopwords=None, join_token=''):
            super().__init__(kill_camel=kill_camel, join_token=join_token)
            if dictionary is not None:
                tokenizer = jieba.Tokenizer(dictionary)
            else:
                tokenizer = jieba.Tokenizer()
            tokenizer.initialize()
            tokenizer.lock = None
            self.tokenizer = tokenizer
            self.stopwords = stopwords

        def cut(self, s: str, cut_all=False, HMM=True, use_paddle=False, **kwargs) -> Iterable[str]:
            if self.kill_camel_pattern is not None:
                s = self.kill_camel(s)
            output = self.tokenizer.cut(s, cut_all=cut_all, HMM=HMM, use_paddle=use_paddle)
            if self.stopwords is not None:
                output = (x for x in output if x not in self.stopwords)
            return output

        def lcut(self, s: str, **kwargs) -> List[str]:
            return list(self.cut(s, **kwargs))

        def __repr__(self):
            return '{}(kill_camel={})'.format(self.__class__.__name__, self.kill_camel_pattern is not None)

    default_tokenizer = JiebaTokenizer(kill_camel=True, stopwords={SEP}, join_token=SEP)
except ImportError:
    default_tokenizer = None
