import json
from all2graph.node import TimeStamp


def test_timestamp():
    a1 = ['2020-01-02', '2020-05-01 12:33:11', 'haha', 'haha']
    a2 = ['2020-01-02', '2020-05-01 12:33:11.11']
    sample_time = '2021-05-01'

    try:
        TimeStamp.from_data(a1, sample_time=sample_time, max_error_rate=0)
        raise RuntimeError('测试max_error_rate失败')
    except AssertionError:
        print('测试max_error_rate成功')

    t1 = TimeStamp.from_data(a1, sample_time=sample_time)
    t2 = TimeStamp.from_data(a2, sample_time=sample_time)
    assert t1 != t2

    t3 = TimeStamp.from_json(json.dumps(t1.to_json()))
    assert t1 == t3

    t4 = TimeStamp.from_data(a1 + a2, sample_time=sample_time)
    t5 = TimeStamp.merge([t1, t2])
    assert t4 == t5
    print(json.dumps(t5.to_json(), indent=4))


if __name__ == '__main__':
    test_timestamp()
    print('测试Timestamp成功')
