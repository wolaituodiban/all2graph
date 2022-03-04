import os
from abc import abstractmethod

import torch
from torch.utils.data import DataLoader

from ..train import Trainer
from ...data import GraphDataset
from ...meta_struct import MetaStruct
from ...parsers import DataParser, GraphParser, PostParser, ParserWrapper


class Model(MetaStruct):
    def __init__(
            self,
            data_parser: DataParser,
            check_point,
            meta_info_configs=None,
            graph_parser_configs=None,
            post_parser: PostParser = None
    ):
        super().__init__(initialized=True)
        self.data_parser = data_parser
        self.check_point = check_point
        os.mkdir(check_point)
        os.mkdir(os.path.join(check_point, 'temp'))
        self.meta_info_configs = meta_info_configs or {}
        self.graph_parser_configs = graph_parser_configs or {}
        self.post_parser = post_parser

        self.meta_info = None
        self.graph_parser = None
        self.trainer = None

    @abstractmethod
    def build_module(self, num_tokens) -> torch.nn.Module:
        raise NotImplementedError

    @property
    def parser_wrapper(self) -> ParserWrapper:
        return ParserWrapper(self.data_parser, self.graph_parser, self.post_parser)

    def fit(self,
            train_data,
            chunksize,
            batch_size,
            loss,
            epoches,
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
        self.meta_info = meta_info or self.data_parser.analyse(
            train_data, chunksize=chunksize, processes=processes, configs=self.meta_info_configs, **kwargs)
        print(self.meta_info)
        self.graph_parser = graph_parser or GraphParser.from_data(self.meta_info, **self.graph_parser_configs)
        print(self.graph_parser)

        # data
        num_workers = processes or os.cpu_count()
        if isinstance(train_data, DataLoader):
            train_dataloader = train_data
        else:
            train_path_df = self.parser_wrapper.save(
                src=train_data,
                dst=os.path.join(self.check_point, 'temp', 'train'),
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
                        dst=os.path.join(self.check_point, 'temp', 'valid_{}'.format(i)),
                        chunksize=chunksize,
                        processes=processes,
                        **kwargs
                    )
                    valid_dataset = GraphDataset(path=valid_path_df, parser=self.parser_wrapper)
                    valid_dataloader = valid_dataset.dataloader(
                        batch_size=batch_size, shuffle=True, num_workers=num_workers)
                    valid_dataloaders.append(valid_dataloader)

        # model
        module = module or self.build_module(self.graph_parser.num_tokens)
        if torch.cuda.is_available():
            module = module.cuda()

        if optimizer_cls is not None:
            optimizer = optimizer_cls(module.parameters(), **(optimizer_kwds or {}))
        else:
            optimizer = None

        self.trainer = Trainer(
            module=module,
            optimizer=optimizer,
            loss=loss,
            data=train_dataloader,
            valid_data=valid_dataloaders,
            metrics=metrics,
            check_point=self.check_point,
            early_stop=early_stop
        )
        print(self.trainer)
        self.trainer.fit(epoches)
