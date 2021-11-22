import sys
import os

import all2graph as ag
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import mean_squared_error


if 'darwin' in sys.platform.lower():
    os.environ['OMP_NUM_THREADS'] = '1'


class TestDataset(Dataset):
    def __init__(self, x, y, key):
        self.x = x
        self.y = y
        self.key = key

    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, item):
        return {self.key: self.x[item]}, {self.key: self.y[item]}


class TestModule(torch.nn.Module):
    def __init__(self, in_feats, out_feats, key):
        super().__init__()
        self.linear1 = torch.nn.Linear(in_features=in_feats, out_features=in_feats)
        self.relu = torch.nn.ReLU()
        self.linear2 = torch.nn.Linear(in_features=in_feats, out_features=out_feats)
        self.key = key

    def forward(self, inputs):
        return {self.key: self.linear2(self.relu(self.linear1(inputs[self.key])))}


def test_trainer():
    num_samples = 100000
    in_feats = 100
    out_feats = 1
    key = 'haha'

    x = torch.randn(num_samples, in_feats, dtype=torch.float32)
    y = torch.randn(num_samples, out_feats, dtype=torch.float32)
    dataset = TestDataset(x, y, key)
    dataloader = DataLoader(dataset, batch_size=64, num_workers=1)
    module = TestModule(in_feats, out_feats, key)

    trainer = ag.nn.Trainer(
        module=module, loss=ag.nn.DictLoss(torch.nn.MSELoss()), data=dataloader,
        metrics={'mse': ag.Metric(mean_squared_error, label_first=True), 'a': ag.Metric(mean_squared_error, label_first=True)},
        valid_data=[dataloader, dataloader],
        early_stop=ag.nn.EarlyStop(1, False, tol=0.01, json_path=ag.json.JsonPathTree([('$.mse', ), ('$.haha', )])),
        check_point=os.path.join(os.path.dirname(__file__), __file__),
    )
    print(trainer)
    trainer.evaluate()
    epochs = 20
    trainer.fit(epochs)
    assert trainer._current_epoch < trainer.train_history.num_epochs < epochs
    trainer = torch.load(os.path.join(trainer.check_point, os.listdir(trainer.check_point)[0]))
    trainer.max_batch = 1000
    trainer.fit(epochs)
    assert trainer._current_epoch < trainer.train_history.num_epochs < epochs
    trainer.early_stop = None
    trainer.fit(epochs)


if __name__ == '__main__':
    test_trainer()
