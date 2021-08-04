import os


def recurse_test_files(_path, _paths):
    if os.path.isdir(_path):
        for _file in os.listdir(_path):
            _file = os.path.join(_path, _file)
            recurse_test_files(_file, _paths)
    elif 'test.py' == _path[-7:]:
        _paths.append(_path)


if __name__ == '__main__':
    paths = []
    recurse_test_files('all2graph', paths)
    for path in paths:
        print(path)
        with open(path, 'r', encoding='UTF-8') as file:
            exec(file.read())
    print('所有测试成功')
