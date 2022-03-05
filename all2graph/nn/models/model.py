import os
import shutil
from abc import abstractmethod

import torch
from torch.utils.data import DataLoader

from ..train import Trainer
from ..utils import Module
from ...data import GraphDataset
from ...parsers import DataParser, GraphParser, PostParser, ParserWrapper


class Model(Module):
    def __init__(
            self,
            check_point=None,
            data_parser: DataParser = None,
            meta_info_configs=None,
            graph_parser_configs=None,
            post_parser: PostParser = None
    ):
        super().__init__()
        self.check_point = check_point
        self.parser_wrapper = ParserWrapper(data_parser=data_parser, post_parser=post_parser)

        self.meta_info = None
        self.meta_info_configs = meta_info_configs or {}
        self.graph_parser_configs = graph_parser_configs or {}
        self.post_parser = post_parser

        self.module = None

    @property
    def data_parser(self):
        return self.parser_wrapper.data_parser

    @data_parser.setter
    def data_parser(self, data_parser):
        self.parser_wrapper.data_parser = data_parser

    @property
    def graph_parser(self):
        return self.parser_wrapper.graph_parser

    @graph_parser.setter
    def graph_parser(self, graph_parser):
        self.parser_wrapper.graph_parser = graph_parser

    @property
    def post_parser(self):
        return self.parser_wrapper.post_parser

    @post_parser.setter
    def post_parser(self, post_parser):
        self.parser_wrapper.post_parser = post_parser

    @property
    def device(self):
        return self.module.device

    @abstractmethod
    def build_module(self, num_tokens) -> torch.nn.Module:
        raise NotImplementedError

    def fit(self,
            train_data,
            loss,
            epoches,
            chunksize=None,
            batch_size=None,
            valid_data: list = None,
            processes=None,
            optimizer_cls=None,
            optimizer_kwds=None,
            metrics=None,
            early_stop=None,
            meta_info=None,
            graph_parser=None,
            module=None,
            **kwargs
            ):
        if graph_parser:
            self.graph_parser = graph_parser
        else:
            self.meta_info = meta_info or self.data_parser.analyse(
                train_data, chunksize=chunksize, processes=processes, configs=self.meta_info_configs, **kwargs)
            print(self.meta_info)
            self.graph_parser = GraphParser.from_data(self.meta_info, **self.graph_parser_configs)
        print(self.graph_parser)

        # data
        if self.check_point:
            if not os.path.exists(self.check_point) or not os.path.isdir(self.check_point):
                os.mkdir(self.check_point)
        num_workers = processes or os.cpu_count()
        if isinstance(train_data, DataLoader):
            train_dataloader = train_data
        else:
            train_path_df = self.parser_wrapper.save(
                src=train_data,
                dst=os.path.join(self.check_point, 'train'),
                chunksize=chunksize,
                processes=processes,
                **kwargs
            )
            train_dataset = GraphDataset(path=train_path_df, parser=self.parser_wrapper)
            train_dataloader = train_dataset.dataloader(batch_size=batch_size, shuffle=True, num_workers=num_workers)

        valid_dataloaders = []
        if valid_data:
            for i, src in enumerate(valid_data):
                if isinstance(src, DataLoader):
                    valid_dataloaders.append(src)
                else:
                    valid_path_df = self.parser_wrapper.save(
                        src=train_data,
                        dst=os.path.join(self.check_point, 'valid_{}'.format(i)),
                        chunksize=chunksize,
                        processes=processes,
                        **kwargs
                    )
                    valid_dataset = GraphDataset(path=valid_path_df, parser=self.parser_wrapper)
                    valid_dataloader = valid_dataset.dataloader(
                        batch_size=batch_size, shuffle=True, num_workers=num_workers)
                    valid_dataloaders.append(valid_dataloader)

        # model
        self.module = module or self.build_module(self.graph_parser.num_tokens)
        if torch.cuda.is_available():
            self.module = module.cuda()

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
