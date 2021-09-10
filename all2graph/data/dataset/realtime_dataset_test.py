import os
import time
import shutil
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
import all2graph as ag
from all2graph.factory import Factory
from all2graph.data.dataset import RealtimeDataset
import platform
if 'darwin' in platform.system().lower():
    os.environ['OMP_NUM_THREADS'] = '1'
path = os.path.dirname(__file__)
path = os.path.dirname(path)
path = os.path.dirname(path)
path = os.path.dirname(path)
csv_path = os.path.join(path, 'test_data', 'MensShoePrices.csv')
to_dir = os.path.join(path, 'test_data', 'chunk')
if os.path.exists(to_dir):
    shutil.rmtree(to_dir)
os.mkdir(to_dir)
ag.split_csv(csv_path, to_dir, chunksize=64, disable=False)

preprocessor = ag.JsonPathTree('json')
json_parser = ag.JsonParser(
    flatten_dict=True, local_index_names={'name'}, segmentation=True
)

factory = Factory(
    data_parser=json_parser, preprocessor=preprocessor, graph_transer_config=dict(
        min_df=0.01, max_df=0.95, top_k=100, top_method='max_tfidf', segment_name=False,
    )
)
processes = os.cpu_count()

factory.produce_meta_graph(
    pd.read_csv(csv_path, nrows=1000), chunksize=int(np.ceil(1000/processes)), progress_bar=True, processes=processes,
)

files = [os.path.join(to_dir, path) for path in os.listdir(to_dir)]


def test_dataset():
    partitions = 3.4
    dataset1 = RealtimeDataset(
        files, factory=factory, disable=False
    )
    num_components1 = 0
    num_meta_nodes1 = 0
    num_meta_edges1 = 0
    for _, labels in ag.progress_wrapper(dataset1):
        assert _[1].ndata[ag.META_NODE_ID].min() == 0
        assert _[0].ndata[ag.META_NODE_ID].max() >= _[1].ndata[ag.META_NODE_ID].max()
        num_components1 += np.unique(_[1].ndata[ag.COMPONENT_ID].abs()).shape[0]
        num_meta_nodes1 += np.unique(_[0].ndata[ag.META_NODE_ID]).shape[0]
        if _[1].num_edges() > 0:
            assert _[1].edata[ag.META_EDGE_ID].min() == 0
            assert _[0].edata[ag.META_EDGE_ID].max() >= _[1].edata[ag.META_EDGE_ID].max()
            num_meta_edges1 += np.unique(_[0].edata[ag.META_EDGE_ID]).shape[0]

    dataset2 = RealtimeDataset(
        files, factory=factory, partitions=partitions, shuffle=True, disable=False
    )
    num_components2 = 0
    num_meta_nodes2 = 0
    num_meta_edges2 = 0
    for _, labels in ag.progress_wrapper(dataset2):
        assert _[1].ndata[ag.META_NODE_ID].min() == 0
        assert _[0].ndata[ag.META_NODE_ID].max() >= _[1].ndata[ag.META_NODE_ID].max()
        num_components2 += np.unique(_[1].ndata[ag.COMPONENT_ID].abs()).shape[0]
        num_meta_nodes2 += np.unique(_[0].ndata[ag.META_NODE_ID]).shape[0]
        if _[1].num_edges() > 0:
            assert _[1].edata[ag.META_EDGE_ID].min() == 0
            assert _[0].edata[ag.META_EDGE_ID].max() >= _[1].edata[ag.META_EDGE_ID].max()
            num_meta_edges2 += np.unique(_[0].edata[ag.META_EDGE_ID]).shape[0]

    assert num_components1 == num_components2 == 10000, (num_components1, num_components2)
    assert num_meta_nodes1 == num_meta_nodes2
    assert num_meta_edges1 == num_meta_edges2
    assert len(dataset1) * int(partitions) == len(dataset2)


def test_data_loader():
    dataset = RealtimeDataset(
        files, factory=factory, partitions=1000, shuffle=True,
        disable=False
    )
    assert len(dataset) == 10000
    num_components1 = 0
    num_meta_nodes1 = 0
    num_meta_edges1 = 0
    start_time1 = time.time()
    for _, labels in ag.progress_wrapper(dataset):
        num_components1 += np.unique(_[1].ndata[ag.COMPONENT_ID].abs()).shape[0]
        num_meta_nodes1 += np.unique(_[0].ndata[ag.META_NODE_ID]).shape[0]
        if _[0].num_edges() > 0:
            num_meta_edges1 += np.unique(_[0].edata[ag.META_EDGE_ID]).shape[0]
    used_time1 = time.time() - start_time1

    data_loader = DataLoader(
        dataset, batch_size=64, shuffle=True, num_workers=os.cpu_count(), pin_memory=True,
        collate_fn=factory.batch
    )
    num_components2 = 0
    num_meta_nodes2 = 0
    num_meta_edges2 = 0
    start_time2 = time.time()
    for _, labels in ag.progress_wrapper(data_loader):
        num_components2 += np.unique(_[1].ndata[ag.COMPONENT_ID].abs()).shape[0]
        num_meta_nodes2 += np.unique(_[0].ndata[ag.META_NODE_ID]).shape[0]
        if _[0].num_edges() > 0:
            num_meta_edges2 += np.unique(_[0].edata[ag.META_EDGE_ID]).shape[0]
    used_time2 = time.time() - start_time2

    assert num_components1 == num_components2 == 10000
    assert num_meta_nodes1 == num_meta_nodes2
    assert num_meta_edges1 == num_meta_edges2
    assert used_time1 > used_time2


if __name__ == '__main__':
    test_dataset()
    test_data_loader()
