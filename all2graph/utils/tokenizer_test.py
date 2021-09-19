from all2graph import JiebaTokenizer, default_tokenizer


def test():
    s = '你好avalonPitch hello!! ！  aadDDa'
    tokenizer = JiebaTokenizer()
    print(tokenizer)
    print(tokenizer.lcut(s))
    print(default_tokenizer)
    print(default_tokenizer.lcut(s))


if __name__ == '__main__':
    test()
