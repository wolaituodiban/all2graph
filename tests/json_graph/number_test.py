from all2graph.json_graph import Number


def test1():
    try:
        Number('test', ecdf=([1], [0.5]))
        raise ValueError('经验分布函数值长度校验功能失败')
    except AssertionError:
        print('经验分布函数值长度校验功能成功')


def test2():
    try:
        Number('test', ecdf=([1, 2], [0]))
        Number('test', ecdf=([1, 2], [1]))
        raise ValueError('经验分布函数值数值范围校验功能失败')
    except AssertionError:
        print('经验分布函数值数值范围校验功能成功')


def test3():
    num = Number('test', ecdf=([1, 2], [0.5]))
    assert num.mean == 1.5, '期望计算验证失败'
    print('期望计算验证成功')


if __name__ == '__main__':
    test1()
    test2()
    test3()
