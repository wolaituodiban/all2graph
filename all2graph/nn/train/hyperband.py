from typing import Union, List

import os
import numpy as np
import torch

from .trainer import Trainer


class HyperBand:
    def __init__(self, trainers: List[Union[Trainer, str]], R, eta=3):
        """
        R: 单个超参能被分配的最大资源
        eta: 淘汰比例
        """
        self.trainers = trainers
        self.R = R
        self.eta = eta

    def __len__(self):
        return len(self.trainers)

    def __getitem__(self, i):
        trainer = self.trainers[i]
        if isinstance(trainer, str):
            file_names = [x for x in os.listdir(trainer) if '.all2graph.trainer' in x]
            file_name = sorted(file_names, key=lambda x: x.split('.')[0])[-1]
            trainer = torch.load(os.path.join(trainer, file_name))
        return trainer

    def top_id(self, n):
        def key(i):
            trainer = self[i]
            best_metric = trainer.best_metric
            if best_metric is None:
                trainer.fit(1)
                best_metric = trainer.best_metric
            return best_metric * trainer.sign

        return sorted(list(range(len(self))), key=key, reverse=True)[:n]

    def run(self):
        rounds = int(np.log(len(self)) / np.log(self.eta)) + 1
        for k in range(rounds):
            n = int(np.ceil(len(self) / self.eta ** k))
            r = int(self.R / self.eta ** (rounds - k - 1))
            for i in self.top_id(n):
                print('hyperband round={}/{}, n={}/{}, r={}/{}, i={}'.format(k + 1, rounds, n, len(self), r, self.R, i))
                trainer = self[i]
                trainer.fit(r - trainer.current_epoch)
