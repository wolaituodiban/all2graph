import sys
import os

import all2graph as ag
import jsonpromax as jpm
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import mean_squared_error


if 'darwin' in sys.platform.lower():
    os.environ['OMP_NUM_THREADS'] = '1'


class TestDataset(Dataset):
    def __init__(self, x, y):
        super().__init__()
        self.x = x
        self.y = y

    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, item):
        return self.x[item], self.y[item]


class TestModule(torch.nn.Module):
    def __init__(self, in_feats, out_feats):
        super().__init__()
        self.linear1 = torch.nn.Linear(in_features=in_feats, out_features=in_feats)
        self.relu = torch.nn.ReLU()
        self.linear2 = torch.nn.Linear(in_features=in_feats, out_features=out_feats)

    def forward(self, inputs):
        return self.linear2(self.relu(self.linear1(inputs)))


def test_trainer():
    num_samples = 100000
    in_feats = 100
    out_feats = 1

    x = torch.randn(num_samples, in_feats, dtype=torch.float32)
    y = torch.randn(num_samples, out_feats, dtype=torch.float32)
    dataset = TestDataset(x, y)
    dataloader = DataLoader(dataset, batch_size=64, num_workers=1)
    module = TestModule(in_feats, out_feats)

    trainer = ag.nn.Trainer(
        module=module, loss=torch.nn.MSELoss(), data=dataloader,
        metrics={'mse': mean_squared_error},
        valid_data=[dataloader, dataloader],
        early_stop=ag.nn.EarlyStop(1, False, tol=0.01, fn=jpm.JsonPathTree([('$.mse',)])),
        check_point=__file__,
        max_history=2,
        save_loader=True
    )
    print(trainer)
    trainer.evaluate()
    epochs = 20
    trainer.fit(epochs)
    assert trainer._current_epoch < trainer.train_history.num_epochs < epochs
    trainer = torch.load(os.path.join(trainer.check_point, os.listdir(trainer.check_point)[0]))
    trainer.save_loader = False
    assert isinstance(trainer, ag.nn.Trainer)
    trainer.max_batch = 1000
    trainer.fit(epochs)
    assert trainer._current_epoch < epochs
    assert trainer.train_history.get_pred(0) is None
    trainer.early_stop = None
    trainer.fit(5)

    # 测试error_msg
    trainer.train_history.loader = None
    trainer.fit(1)
    assert trainer.error_msg is not None
    print('success')


if __name__ == '__main__':
    test_trainer()
