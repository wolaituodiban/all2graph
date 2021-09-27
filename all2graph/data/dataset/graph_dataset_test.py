import os
import shutil
from torch.utils.data import DataLoader
import all2graph as ag
from all2graph import JsonParser, Timer
from all2graph.data.dataset import GraphDataset
from all2graph.factory import Factory

import platform
if 'darwin' in platform.system().lower():
    os.environ['OMP_NUM_THREADS'] = '1'


def test_graph_file():
    path = os.path.dirname(__file__)
    path = os.path.dirname(path)
    path = os.path.dirname(path)
    path = os.path.dirname(path)
    csv_path = os.path.join(path, 'test_data', 'MensShoePrices.csv')
    nrows = 1000

    json_parser = JsonParser(
        'json', flatten_dict=True, local_id_keys={'name'}, segment_value=True
    )

    factory = Factory(data_parser=json_parser)

    factory.analyse(
        csv_path, chunksize=64, progress_bar=True, processes=None, nrows=nrows
    )

    # 测试保存文件
    save_path = os.path.join(path, 'test_data', 'graphs')
    if os.path.exists(save_path):
        shutil.rmtree(save_path)
    os.mkdir(save_path)
    ag.split_csv(csv_path, save_path, chunksize=64, disable=False, zip=True)
    graph_paths = [os.path.join(save_path, file) for file in os.listdir(save_path)]

    # 开始测试
    with Timer('dataloader'):
        dataset3 = GraphDataset(graph_paths, factory=factory, partitions=1000, shuffle=True, disable=False)

        data_loader1 = DataLoader(
            dataset3, batch_size=64, shuffle=True, num_workers=os.cpu_count(), pin_memory=True,
            collate_fn=dataset3.collate_fn
        )
        for (meta_graph, graph), labels in ag.progress_wrapper(data_loader1):
            pass

    shutil.rmtree(save_path)


if __name__ == '__main__':
    test_graph_file()
