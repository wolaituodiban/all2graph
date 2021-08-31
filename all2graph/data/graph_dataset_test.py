import os
from all2graph.data import GraphDataset


path = os.path.join('..', '..', 'test_data', 'graphs')


def test():
    graph_paths = [os.path.join(path, file) for file in os.listdir(path)]
    dataset = GraphDataset(graph_paths, partitions=2, disable=False)
    print(len(dataset))


if __name__ == '__main__':
    test()
