import os
import numpy as np
import shutil
from torch.utils.data import DataLoader
import all2graph as ag
from all2graph import JsonParser, Timer, default_tokenizer
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
        'json', flatten_dict=True, local_index_names={'name'}, segment_value=True
    )

    factory = Factory(data_parser=json_parser)

    factory.produce_meta_graph(
        csv_path, chunksize=64, progress_bar=True, processes=None, nrows=nrows
    )

    # 测试保存文件
    save_path = os.path.join(path, 'test_data', 'graphs')
    if os.path.exists(save_path):
        shutil.rmtree(save_path)
    os.mkdir(save_path)
    factory.save_graphs(csv_path, save_path, chunksize=64, progress_bar=True, processes=None, nrows=nrows)
    graph_paths = [os.path.join(save_path, file) for file in os.listdir(save_path)]

    # 开始测试
    partitions = 3.4
    dataset1 = GraphDataset(graph_paths, factory=factory, disable=False)
    num_components1 = 0
    num_meta_nodes1 = 0
    num_meta_edges1 = 0
    for _, labels in ag.progress_wrapper(dataset1):
        assert _[1].ndata[ag.META_NODE_ID].min() == 0
        assert _[0].ndata[ag.META_NODE_ID].max() == _[1].ndata[ag.META_NODE_ID].max(), (
            _[0].ndata[ag.META_NODE_ID].max(), _[1].ndata[ag.META_NODE_ID].max())
        num_components1 += np.unique(_[0].ndata[ag.COMPONENT_ID]).shape[0]
        num_meta_nodes1 += np.unique(_[0].ndata[ag.META_NODE_ID]).shape[0]
        if _[1].num_edges() > 0:
            assert _[1].edata[ag.META_EDGE_ID].min() == 0
            assert _[0].edata[ag.META_EDGE_ID].max() == _[1].edata[ag.META_EDGE_ID].max()
            num_meta_edges1 += np.unique(_[0].edata[ag.META_EDGE_ID]).shape[0]

    dataset2 = GraphDataset(graph_paths, factory=factory, partitions=partitions, shuffle=True, disable=False)
    num_components2 = 0
    num_meta_nodes2 = 0
    num_meta_edges2 = 0
    for _, labels in ag.progress_wrapper(dataset2):
        assert _[1].ndata[ag.META_NODE_ID].min() >= 0, _[1].ndata[ag.META_NODE_ID].min()
        assert _[0].ndata[ag.META_NODE_ID].max() == _[1].ndata[ag.META_NODE_ID].max()
        num_components2 += np.unique(_[0].ndata[ag.COMPONENT_ID]).shape[0]
        num_meta_nodes2 += np.unique(_[0].ndata[ag.META_NODE_ID]).shape[0]
        if _[1].num_edges() > 0:
            assert _[1].edata[ag.META_EDGE_ID].min() >= 0, _[1].edata[ag.META_EDGE_ID].min()
            assert _[0].edata[ag.META_EDGE_ID].max() == _[1].edata[ag.META_EDGE_ID].max()
            num_meta_edges2 += np.unique(_[0].edata[ag.META_EDGE_ID]).shape[0]

    assert len(dataset1) * int(partitions) == len(dataset2), (len(dataset1), len(dataset2))
    assert num_components1 == num_components2 == nrows, (num_components1, num_components2, nrows)
    assert num_meta_nodes1 == num_meta_nodes2
    assert num_meta_edges1 == num_meta_edges2

    dataset3 = GraphDataset(graph_paths, factory=factory, partitions=1000, shuffle=True, disable=False)
    assert len(dataset3) == nrows
    num_components1 = 0
    num_meta_nodes1 = 0
    num_meta_edges1 = 0
    with Timer('dataset') as timer:
        for _, labels in ag.progress_wrapper(dataset3):
            num_components1 += np.unique(_[0].ndata[ag.COMPONENT_ID]).shape[0]
            num_meta_nodes1 += np.unique(_[0].ndata[ag.META_NODE_ID]).shape[0]
            if _[0].num_edges() > 0:
                num_meta_edges1 += np.unique(_[0].edata[ag.META_EDGE_ID]).shape[0]
        used_time1 = timer.diff()

    data_loader = DataLoader(
        dataset3, batch_size=64, shuffle=True, num_workers=os.cpu_count(), pin_memory=True,
        collate_fn=Factory.batch
    )
    num_components2 = 0
    num_meta_nodes2 = 0
    num_meta_edges2 = 0
    with Timer('dataloader') as timer:
        for _, labels in ag.progress_wrapper(data_loader):
            num_components2 += np.unique(_[0].ndata[ag.COMPONENT_ID]).shape[0]
            num_meta_nodes2 += np.unique(_[0].ndata[ag.META_NODE_ID]).shape[0]
            if _[0].num_edges() > 0:
                num_meta_edges2 += np.unique(_[0].edata[ag.META_EDGE_ID]).shape[0]
        used_time2 = timer.diff()

    assert num_components1 == num_components2 == nrows
    assert num_meta_nodes1 == num_meta_nodes2
    assert num_meta_edges1 == num_meta_edges2
    assert used_time1 > used_time2

    shutil.rmtree(save_path)


if __name__ == '__main__':
    test_graph_file()
