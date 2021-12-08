import time
import random
import all2graph as ag
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


class TestDataset(Dataset):
    def __init__(self, length):
        self.length = length

    def __len__(self):
        return self.length

    def __getitem__(self, item):
        worker_id = torch.utils.data.get_worker_info().id
        return worker_id, item


def test_partition_sampler():
    def get_partition(i):
        for j, ind in enumerate(indices):
            if i in ind:
                return j

    lines = 0
    indices = []
    for x in [45, 11, 26, 23, 42, 22, 53]:
        indices.append(list(range(lines, lines + x)))
        lines += x
    num_workers = 2

    sampler = ag.data.PartitionSampler(indices=indices, num_workers=num_workers, shuffle=True)
    dataset = TestDataset(lines)
    data_loader = DataLoader(dataset, num_workers=num_workers, sampler=sampler)

    for worker_id, ind in data_loader:
        print(worker_id, ind, get_partition(ind))


if __name__ == '__main__':
    test_partition_sampler()
