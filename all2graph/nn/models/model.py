import os
import shutil
from abc import abstractmethod
from typing import Dict

import pandas as pd
import torch
from torch.utils.data import DataLoader

from ..train import Trainer
from ..utils import Module, predict_csv
from ...data import GraphDataset
from ...info import MetaInfo
from ...parsers import DataParser, GraphParser, PostParser, ParserWrapper


class Model(Module):
    def __init__(
            self,
            module: torch.nn.Module = None,
            parser: ParserWrapper = None,
            data_parser: DataParser = None,
            graph_parser: GraphParser = None,
            post_parser: PostParser = None,
            meta_info: MetaInfo = None,
            meta_info_configs=None,
            graph_parser_configs=None,
            check_point=None,
    ):
        super().__init__()
        self.module = module
        parser = parser or ParserWrapper(data_parser=data_parser, graph_parser=graph_parser, post_parser=post_parser)
        self.parser = parser

        self.meta_info = meta_info
        self.meta_info_configs = meta_info_configs or {}
        self.graph_parser_configs = graph_parser_configs or {}
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
    def post_parser(self):
        return self.parser.post_parser

    @post_parser.setter
    def post_parser(self, post_parser):
        self.parser.post_parser = post_parser

    @property
    def device(self):
        return self.module.device

    @abstractmethod
    def build_module(self, num_tokens) -> torch.nn.Module:
        raise NotImplementedError

    def build_data(self, train_data, batch_size, valid_data=None, processes=None, **kwargs):
        # MetaInfo
        if self.meta_info is None:
            self.meta_info = self.data_parser.analyse(
                train_data, processes=processes, configs=self.meta_info_configs, **kwargs)
        print(self.meta_info)

        # GraphParser
        if self.graph_parser is None:
            self.graph_parser = GraphParser.from_data(self.meta_info, **self.graph_parser_configs)
        print(self.graph_parser)

        # DataLoader
        num_workers = processes or os.cpu_count()
        if isinstance(train_data, DataLoader):
            train_dataloader = train_data
        else:
            train_path_df = self.parser.save(
                src=train_data,
                dst=os.path.join(self.check_point, 'train'),
                processes=processes,
                **kwargs
            )
            train_dataset = GraphDataset(path=train_path_df, parser=self.parser)
            train_dataloader = train_dataset.dataloader(batch_size=batch_size, shuffle=True, num_workers=num_workers)

        valid_dataloaders = []
        if valid_data:
            for i, src in enumerate(valid_data):
                if isinstance(src, DataLoader):
                    valid_dataloaders.append(src)
                else:
                    valid_path_df = self.parser.save(
                        src=train_data,
                        dst=os.path.join(self.check_point, 'valid_{}'.format(i)),
                        processes=processes,
                        **kwargs
                    )
                    valid_dataset = GraphDataset(path=valid_path_df, parser=self.parser)
                    valid_dataloader = valid_dataset.dataloader(
                        batch_size=batch_size, shuffle=True, num_workers=num_workers)
                    valid_dataloaders.append(valid_dataloader)
        return train_dataloader, valid_dataloaders

    def fit(self,
            train_data,
            loss,
            epoches,
            batch_size=None,
            valid_data: list = None,
            processes=None,
            optimizer_cls=None,
            optimizer_kwds=None,
            metrics=None,
            early_stop=None,
            **kwargs
            ):
        if self.check_point:
            if not os.path.exists(self.check_point) or not os.path.isdir(self.check_point):
                os.mkdir(self.check_point)

        # data
        train_dataloader, valid_dataloaders = self.build_data(
            train_data=train_data, batch_size=batch_size, valid_data=valid_data, processes=processes, **kwargs)

        # model
        if self.module is None:
            self.module = self.build_module(self.graph_parser.num_tokens)
        if torch.cuda.is_available():
            self.module.cuda()

        if optimizer_cls is not None:
            optimizer = optimizer_cls(self.module.parameters(), **(optimizer_kwds or {}))
        else:
            optimizer = None

        trainer = Trainer(
            module=self.module,
            optimizer=optimizer,
            loss=loss,
            data=train_dataloader,
            valid_data=valid_dataloaders,
            metrics=metrics,
            check_point=self.check_point,
            early_stop=early_stop
        )
        print(trainer)
        trainer.fit(epoches)
        shutil.rmtree(self.check_point)

    def predict(self, src, **kwargs):
        return predict_csv(self.parser, self.module, src, **kwargs)

    @torch.no_grad()
    def forward(self, df: pd.DataFrame) -> Dict[str, torch.Tensor]:
        self.eval()
        graph = self.parser(df)
        return self.module(graph)
