from typing import List

import numpy as np
from torch.utils.data import Sampler


class PartitionSampler(Sampler):
    def __init__(self, indices: List[List[int]], num_workers: int, shuffle=False, batch_size=1):
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
        self.batch_size = batch_size

    def __len__(self):
        """
        由于一下原因，我们无法得到每次迭代时的精确长度：
        1、文件数量不一定是worker数量的整数倍
        2、每个文件的长度不一定相同
        3、shuffle带来的随机性
        因此，我们只提供一个迭代器长度的上确界，
        在任何情况下，迭代器的实际长度都小于这个上确界
        Returns:

        """
        num_samples = max(max(ind) for ind in self.indices)
        batch_per_worker = np.ceil(num_samples / self.num_workers / self.batch_size)
        return int(batch_per_worker * self.num_workers)

    def __iter__(self):
        if self.shuffle:
            np.random.shuffle(self.indices)
            for ind in self.indices:
                np.random.shuffle(ind)

        # 丢掉不能被整分的partition
        remainder = len(self.indices) % self.num_workers
        indices = self.indices[remainder:]

        # 依次将每个partition分配给worker
        worker_indices = [[] for _ in range(self.num_workers)]
        for i, ind in enumerate(indices):
            worker_indices[i % self.num_workers] += ind

        # 根据最短的worker_indices的长度进行剪裁
        # 将worker_indices间隔穿插成最终的indices
        batch_per_worker = int(np.ceil(min(map(len, worker_indices)) / self.batch_size))
        for i_batch in range(batch_per_worker):
            for i_worker in range(self.num_workers):
                yield worker_indices[i_worker][(i_batch * self.batch_size):((i_batch + 1) * self.batch_size)]
