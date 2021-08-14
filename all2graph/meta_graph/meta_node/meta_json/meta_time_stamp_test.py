import json
from all2graph.meta_graph.meta_node import MetaTimeStamp


def test_timestamp():
    a1 = ['2020-01-02', '2020-05-01 12:33:11']
    a2 = ['2020-01-02', '2020-05-03 12:01:23.11']
    sample_time = '2021-05-01'

    t1 = MetaTimeStamp.from_data(len(a1), sample_ids=list(range(len(a1))), values=a1, sample_times=sample_time)
    t2 = MetaTimeStamp.from_data(len(a2), sample_ids=list(range(len(a2))), values=a2, sample_times=sample_time)
    assert t1 != t2

    t3 = MetaTimeStamp.from_json(json.dumps(t1.to_json()))
    assert t1 == t3, '{}\n{}'.format(t1.to_json(), t3.to_json())
    a3 = a1 + a2
    t4 = MetaTimeStamp.from_data(len(a3), sample_ids=list(range(len(a3))), values=a3, sample_times=sample_time)
    t5 = MetaTimeStamp.reduce([t1, t2])
    assert t4 == t5
    print(json.dumps(t5.to_json(), indent=4))


if __name__ == '__main__':
    test_timestamp()
    print('测试Timestamp成功')
