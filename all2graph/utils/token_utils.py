from functools import lru_cache
import jieba


class SpecialToken:
    pass

class CLS(SpecialToken):
    pass

class UNKNOWN(SpecialToken):
    pass
    
class END(SpecialToken):
    pass

class PADDING(SpecialToken):
    pass

class SEP(SpecialToken):
    pass


class Tokenizer:
    def __init__(self, stopwords=['', ' ', '_']):
        tokenizer = jieba.Tokenizer()
        tokenizer.initialize()
        # 把锁删掉，防止多进程dataloader卡住
        tokenizer.lock = None
        self.tokenizer = tokenizer
        self.stopwords = set(stopwords)
    
    @lru_cache(1024)
    def lcut(self, s: str):
        output = []
        for token in self.tokenizer.cut(s):
            if token in self.stopwords:
                continue
            try:
                float(token)
                output.extend(token)
            except:
                output.append(token)
        return output


tokenizer = Tokenizer()