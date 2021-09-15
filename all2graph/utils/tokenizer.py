from abc import abstractmethod
from typing import List, Iterable
import re


class Tokenizer:
    def __init__(self, kill_camel):
        if kill_camel:
            self.kill_camel_pattern = re.compile(r'([a-z])([A-Z])')
        else:
            self.kill_camel_pattern = None

    def kill_camel(self, s):
        return self.kill_camel_pattern.sub(r'\1 \2', s)

    def cut(self, s: str, **kwargs) -> Iterable[str]:
        raise NotImplementedError

    def lcut(self, s: str, **kwargs) -> List[str]:
        raise NotImplementedError

    @abstractmethod
    def join(self, x: List[str], **kwargs) -> str:
        return ''.join(x)


class JiebaTokenizer(Tokenizer):
    def __init__(self, kill_camel=False, dictionary=None):
        super().__init__(kill_camel=kill_camel)
        import jieba
        if dictionary is not None:
            tokenizer = jieba.Tokenizer(dictionary)
        else:
            tokenizer = jieba.Tokenizer()
        tokenizer.initialize()
        tokenizer.lock = None
        self.tokenizer = tokenizer

    def cut(self, s: str, cut_all=False, HMM=True, use_paddle=False, **kwargs) -> Iterable[str]:
        if self.kill_camel_pattern is not None:
            s = self.kill_camel(s)
        return self.tokenizer.cut(s, cut_all=cut_all, HMM=HMM, use_paddle=use_paddle)

    def lcut(self, s: str, **kwargs) -> List[str]:
        return list(self.cut(s, **kwargs))

    def join(self, x: List[str], **kwargs) -> str:
        return super().join(x, **kwargs)


default_tokenizer = JiebaTokenizer(kill_camel=True)
