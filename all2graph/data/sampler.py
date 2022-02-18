from copy import deepcopy
from typing import List

import numpy as np
from torch.utils.data import Sampler


class PartitionSampler(Sampler):
    """保证读取文件的复杂度为O(N)"""
    def __init__(self, indices: List[List[int]], num_workers: int, shuffle=False, batch_size=1):
        """

        Args:
            indices: list of list of indices
            num_workers: DataLoader的worker数量，会根据这个值将同一个partition分发给同一个workder
            shuffle: 是否shuffle
        """
        super().__init__(None)
        self.indices = deepcopy(indices)
        self.num_workers = max(num_workers, 1)
        self.shuffle = shuffle
        self.batch_size = batch_size

    @property
    def num_samples(self):
        return max(max(ind) for ind in self.indices)

    def __len__(self):
        return int(np.ceil(self.num_samples / self.batch_size))

    def __iter__(self):
        # 在保持每个partition的id都是连续的情况下，将所有id组成一个list
        if self.shuffle:
            np.random.shuffle(self.indices)
            for ind in self.indices:
                np.random.shuffle(ind)
        all_indices = []
        for ind in self.indices:
            all_indices += ind

        # 计算每个worker的batch数量
        all_num_batch = len(self)
        batch_per_worker = int(all_num_batch / self.num_workers)
        num_batch_diff = all_num_batch - batch_per_worker * self.num_workers
        num_batches = [batch_per_worker + 1] * num_batch_diff + [batch_per_worker] * (self.num_workers - num_batch_diff)
        assert sum(num_batches) == all_num_batch

        # 依次将idx分配给worker
        worker_indices = []
        for num_batch in num_batches:
            num_sample = num_batch*self.batch_size
            indices = all_indices[:num_sample]
            all_indices = all_indices[num_sample:]
            worker_indices.append(indices)

        # 每个worker交替生成index
        for _ in range(max(num_batches)):
            for i, indices in enumerate(worker_indices):
                if len(indices) == 0:
                    continue
                yield indices[:self.batch_size]
                worker_indices[i] = indices[self.batch_size:]
