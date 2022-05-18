import pandas as pd
import torch
import numpy as np

from ..framework import Framework
from ..loss import DictLoss
from ..train import Trainer
from ..utils import Module, predict_csv
from ...data import CSVDataset
from ...info import MetaInfo
from ...parsers import DataParser, GraphParser, ParserWrapper
from ...utils import Metric


class Model(Module):
    def __init__(
            self,
            data_parser: DataParser = None,
            meta_info_configs=None,
            meta_info: MetaInfo = None,
            graph_parser_configs=None,
            graph_parser: GraphParser = None,
            parser: ParserWrapper = None,
            module: Framework = None,
            check_point=None,
    ):
        super().__init__()
        self.meta_info_configs = meta_info_configs or {}
        self.meta_info = meta_info
        self.graph_parser_configs = graph_parser_configs or {}
        parser = parser or ParserWrapper(data_parser=data_parser, graph_parser=graph_parser)
        self.parser = parser
        self.module = module
        self.check_point = check_point

    @property
    def data_parser(self):
        return self.parser.data_parser

    @data_parser.setter
    def data_parser(self, data_parser):
        self.parser.data_parser = data_parser

    @property
    def graph_parser(self):
        return self.parser.graph_parser

    @graph_parser.setter
    def graph_parser(self, graph_parser):
        self.parser.graph_parser = graph_parser

    @property
    def device(self):
        return self.module.device

    def forward(self, inputs: pd.DataFrame):
        if isinstance(inputs, pd.DataFrame):
            self.eval()
            inputs = self.parser(inputs)
        if isinstance(inputs, dict):
            return {k: self.module(v) for k, v in inputs.items()}
        else:
            return self.module(inputs)

    def build_module(self):
        raise NotImplementedError

    def fit(self,
            train_data,
            epoches,
            batch_size,
            loss,
            chunksize=None,
            valid_data: list = None,
            processes=0,
            optimizer_cls=None,
            optimizer_kwds=None,
            metrics=None,
            early_stop=None,
            analyse_frac=None,
            pin_memory=False,
            **kwargs
            ):
        """

        Args:
            train_data: dataframe, 长度与样本数量相同，包含一列path代表每个样本的文件地址
            loss:
            epoches:
            batch_size:
            valid_data:
            processes:
            optimizer_cls:
            optimizer_kwds:
            metrics:
            early_stop:
            analyse_frac: 分析阶段的数据采样率
            pin_memory:
            **kwargs:

        Returns:

        """
        # 检查parser是否完全
        chunksize = chunksize or batch_size
        if not isinstance(loss, DictLoss):
            loss = DictLoss(loss)
        if metrics is not None:
            metrics = {k: v if isinstance(v, Metric) else Metric(v, label_first=False) for k, v in metrics.items()}
        assert self.data_parser is not None, 'please set data_parser first'
        if self.graph_parser is None:
            print('graph_parser not set, start building')
            if self.meta_info is None:
                print('meta_info not set, start building')
                paths = train_data['path'].unique()
                if analyse_frac is not None:
                    paths = np.random.choice(paths, int(analyse_frac*paths.shape[0]), replace=False)
                self.meta_info = self.data_parser.analyse(
                    paths, processes=processes, configs=self.meta_info_configs, chunksize=chunksize, **kwargs)
                print(self.meta_info)
            self.graph_parser = GraphParser.from_data(self.meta_info, **self.graph_parser_configs)
            print(self.graph_parser)
        assert isinstance(self.graph_parser, GraphParser), 'fitting not support multi-parsers'

        # dataloader
        train_data = CSVDataset(train_data, self.parser, **kwargs).dataloader(
            batch_size=batch_size, num_workers=processes, shuffle=True, pin_memory=pin_memory)

        if valid_data is not None:
            valid_data = [
                CSVDataset(x, self.parser, **kwargs).dataloader(
                    batch_size=batch_size, num_workers=processes, shuffle=True, pin_memory=pin_memory)
                for x in valid_data
            ]

        # build module
        if self.module is None:
            self.build_module()

        # train
        if optimizer_cls is not None:
            optimizer = optimizer_cls(self.module.parameters(), **(optimizer_kwds or {}))
        else:
            optimizer = None

        trainer = Trainer(
            module=self,
            optimizer=optimizer,
            loss=loss,
            data=train_data,
            valid_data=valid_data,
            metrics=metrics,
            check_point=self.check_point,
            early_stop=early_stop
        )
        print(trainer)
        trainer.fit(epoches)

    def predict(self, src, **kwargs):
        return predict_csv(self.parser, self.module, src, **kwargs)


