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

    @staticmethod
    def camel_to_snake(s):
        up_index = []
        for i, c in enumerate(s):
            if c.isupper():
                up_index.append(i)  # 获取大写字符索引位置
        ls = s.lower()  # 原字符串转小写
        # print(ls)
        list_ls = list(ls)  # 转列表
        if up_index:
            addi = 0
            for g in up_index:
                list_ls.insert(g + addi, '_')  # 插入_
                addi += 1
        last_ls = ''.join(list_ls)  # 转回字符
        # print(last_ls)
        return last_ls

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
