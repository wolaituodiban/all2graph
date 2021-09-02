import os
import time
import shutil
import pickle
import numpy as np
from torch.utils.data import DataLoader
import all2graph as ag
from all2graph.data.dataset import RealtimeDataset
import platform
if 'darwin' in platform.system().lower():
    os.environ['OMP_NUM_THREADS'] = '1'
path = os.path.dirname(__file__)
path = os.path.dirname(path)
path = os.path.dirname(path)
path = os.path.dirname(path)
to_dir = os.path.join(path, 'test_data', 'chunk')
if os.path.exists(to_dir):
    shutil.rmtree(to_dir)
os.mkdir(to_dir)
ag.split_csv(
    os.path.join(path, 'test_data', 'MensShoePrices.csv'),
    to_dir,
    chunksize=64,
    disable=False
)

preprocessor = ag.JsonPathTree('json')
json_parser = ag.JsonParser(
    root_name=preprocessor.json_col, flatten_dict=True, local_index_names={'name'}, segmentation=True
)

with open(os.path.join(path, 'test_data/graph_trans.pkl'), 'br') as file:
    transer = pickle.load(file)

files = [os.path.join(to_dir, path) for path in os.listdir(to_dir)]


def test_dataset():
    partitions = 3.4
    dataset1 = RealtimeDataset(
        files, preprocessor=preprocessor, data_parser=json_parser, graph_transer=transer, disable=False
    )
    num_components1 = 0
    num_meta_nodes1 = 0
    num_meta_edges1 = 0
    for _ in ag.progress_wrapper(dataset1):
        assert _[1].ndata[ag.META_NODE_IDS].min() == 0
        assert _[0].ndata[ag.META_NODE_IDS].max() >= _[1].ndata[ag.META_NODE_IDS].max()
        num_components1 += np.unique(_[1].ndata[ag.COMPONENT_IDS]).shape[0]
        num_meta_nodes1 += np.unique(_[0].ndata[ag.META_NODE_IDS]).shape[0]
        if _[1].num_edges() > 0:
            assert _[1].edata[ag.META_EDGE_IDS].min() == 0
            assert _[0].edata[ag.META_EDGE_IDS].max() >= _[1].edata[ag.META_EDGE_IDS].max()
            num_meta_edges1 += np.unique(_[0].edata[ag.META_EDGE_IDS]).shape[0]

    dataset2 = RealtimeDataset(
        files, preprocessor=preprocessor, data_parser=json_parser, graph_transer=transer,
        partitions=partitions, shuffle=True, disable=False
    )
    num_components2 = 0
    num_meta_nodes2 = 0
    num_meta_edges2 = 0
    for _ in ag.progress_wrapper(dataset2):
        assert _[1].ndata[ag.META_NODE_IDS].min() == 0
        assert _[0].ndata[ag.META_NODE_IDS].max() >= _[1].ndata[ag.META_NODE_IDS].max()
        num_components2 += np.unique(_[1].ndata[ag.COMPONENT_IDS]).shape[0]
        num_meta_nodes2 += np.unique(_[0].ndata[ag.META_NODE_IDS]).shape[0]
        if _[1].num_edges() > 0:
            assert _[1].edata[ag.META_EDGE_IDS].min() == 0
            assert _[0].edata[ag.META_EDGE_IDS].max() >= _[1].edata[ag.META_EDGE_IDS].max()
            num_meta_edges2 += np.unique(_[0].edata[ag.META_EDGE_IDS]).shape[0]

    assert num_components1 == num_components2 == 10000, (num_components1, num_components2)
    assert num_meta_nodes1 == num_meta_nodes2
    assert num_meta_edges1 == num_meta_edges2
    assert len(dataset1) * int(partitions) == len(dataset2)


def test_data_loader():
    dataset = RealtimeDataset(
        files, preprocessor=preprocessor, data_parser=json_parser, graph_transer=transer, partitions=1000, shuffle=True,
        disable=False
    )
    assert len(dataset) == 10000
    num_components1 = 0
    num_meta_nodes1 = 0
    num_meta_edges1 = 0
    start_time1 = time.time()
    for _ in ag.progress_wrapper(dataset):
        num_components1 += np.unique(_[1].ndata[ag.COMPONENT_IDS]).shape[0]
        num_meta_nodes1 += np.unique(_[0].ndata[ag.META_NODE_IDS]).shape[0]
        if _[0].num_edges() > 0:
            num_meta_edges1 += np.unique(_[0].edata[ag.META_EDGE_IDS]).shape[0]
    used_time1 = time.time() - start_time1

    data_loader = DataLoader(
        dataset, batch_size=64, shuffle=True, num_workers=os.cpu_count(), pin_memory=True,
        collate_fn=dataset.collate_fn
    )
    num_components2 = 0
    num_meta_nodes2 = 0
    num_meta_edges2 = 0
    start_time2 = time.time()
    for _ in ag.progress_wrapper(data_loader):
        num_components2 += np.unique(_[1].ndata[ag.COMPONENT_IDS]).shape[0]
        num_meta_nodes2 += np.unique(_[0].ndata[ag.META_NODE_IDS]).shape[0]
        if _[0].num_edges() > 0:
            num_meta_edges2 += np.unique(_[0].edata[ag.META_EDGE_IDS]).shape[0]
    used_time2 = time.time() - start_time2

    assert num_components1 == num_components2 == 10000
    assert num_meta_nodes1 == num_meta_nodes2
    assert num_meta_edges1 == num_meta_edges2
    assert used_time1 > used_time2


if __name__ == '__main__':
    test_dataset()
    test_data_loader()
