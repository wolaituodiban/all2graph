import time
import random
import all2graph as ag
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


class ShuffleDataset(Dataset):
    def __init__(self):
        self.data = np.arange(0, 1000)
        self._shuffled = False

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, item):
        time.sleep(random.random())
        worker_id = torch.utils.data.get_worker_info().id
        return worker_id, item


if __name__ == '__main__':
    dataset = ShuffleDataset()
    data_loader = DataLoader(dataset, num_workers=4, shuffle=False, prefetch_factor=1)
    with ag.Timer('haha'):
        for x in data_loader:
            print(x)
