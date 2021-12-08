from typing import List

import numpy as np
from torch.utils.data import Sampler


class PartitionSampler(Sampler):
    def __init__(self, indices: List[List[int]], num_workers: int, shuffle=False):
        """

        Args:
            indices: list of list of indices
            num_workers: DataLoader的worker数量，会根据这个值将同一个partition分发给同一个workder
            shuffle: 是否shuffle
        """
        super().__init__(None)
        self.indices = indices
        self.num_workers = num_workers
        self.shuffle = shuffle

    def __len__(self):
        return max(max(ind) for ind in self.indices)

    def __iter__(self):
        if self.shuffle:
            np.random.shuffle(self.indices)
            for ind in self.indices:
                np.random.shuffle(ind)

        # 丢掉不能被整分的partition
        remainder = len(self.indices) % self.num_workers
        indices = self.indices[:-remainder]

        # 依次将每个partition分配给worker
        worker_indices = [[] for _ in range(self.num_workers)]
        for i, ind in enumerate(indices):
            worker_indices[i % self.num_workers] += ind

        # 根据最短的worker_indices的长度进行剪裁
        # 将worker_indices间隔穿插成最终的indices
        min_len = min(map(len, worker_indices))
        for j in range(min_len):
            for i in range(self.num_workers):
                yield worker_indices[i][j]
