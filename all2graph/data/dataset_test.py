import os
import shutil

import pandas as pd
import all2graph as ag
from all2graph import JsonParser, Timer
from all2graph.data import CSVDataset

import platform
if 'darwin' in platform.system().lower():
    os.environ['OMP_NUM_THREADS'] = '1'


def test_graph_file():
    path = os.path.dirname(__file__)
    path = os.path.dirname(path)
    path = os.path.dirname(path)
    csv_path = os.path.join(path, 'test_data', 'MensShoePrices.csv')
    df = pd.read_csv(csv_path, nrows=1000)

    json_parser = JsonParser(
        'json', flatten_dict=True, local_id_keys={'name'}, segment_value=True
    )
    raw_graph_parser = ag.RawGraphParser.from_data(
        ag.MetaInfo.from_data(json_parser.parse(df, progress_bar=True)[0], progress_bar=True)
    )
    # 测试保存文件
    save_path = os.path.join(path, 'test_data', 'temp')
    ag.split_csv(df, save_path, chunksize=5, disable=False, zip=True, concat_chip=True)
    graph_paths = [os.path.join(save_path, file) for file in os.listdir(save_path)]

    # 开始测试
    with Timer('dataset'):
        dataset = CSVDataset(
            graph_paths, data_parser=json_parser, raw_graph_parser=raw_graph_parser,
            chunksize=32, shuffle=True, disable=False)
        num_rows2 = []
        for graph, labels in ag.progress_wrapper(dataset):
            num_rows2.append(graph.num_components)
    shutil.rmtree(save_path)
    temp = []
    for a in dataset.paths:
        for b, c in a.items():
            for d in c:
                temp.append((b, d))
    # 测试每一个样本不重复不遗漏
    assert len(set(temp)) == df.shape[0]
    assert df.shape[0] == sum(num_rows2)
    # 测试batchsize正确
    assert set(num_rows2[:-1]) == {dataset.chunksize}, (num_rows2, dataset.chunksize)


if __name__ == '__main__':
    test_graph_file()
