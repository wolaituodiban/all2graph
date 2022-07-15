"""模型封装"""
import pandas as pd
import numpy as np
import torch
from torch.nn.functional import cross_entropy
from torch.utils.data import DataLoader

from ..framework import Framework
from ..train import Trainer
from ..utils import Module, predict_csv
from ...data import CSVDataset
from ...graph import Graph
from ...info import MetaInfo
from ...parsers import DataParser, GraphParser, ParserWrapper


MASK_LOSS = '__mask_loss'
MASK_ACCURACY = '__mask_accuracy'


class MaskLossWrapper(Module):
    """
    融合原本的loss和mask任务的loss
    Args:
        loss: 原本的loss函数
        weight: mask loss的权重
    """
    def __init__(self, loss, weight=1):
        super().__init__()
        self.loss = loss
        self.weight = weight

    def forward(self, inputs, target):
        mask_loss = inputs[MASK_LOSS] * self.weight
        if self.loss is not None:
            return self.loss(inputs, target) + mask_loss
        return mask_loss


def get_mask_loss(_, inputs):
    return inputs[MASK_LOSS].mean()


def get_mask_accuracy(_, inputs):
    return inputs[MASK_ACCURACY].mean()


class Model(Module):
    """模型封装基类, 定义fit方法"""
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
            mask_prob=0,
            mask_loss_weight=1,
            return_details=False,
            return_embedding=False
    ):
        super().__init__()
        self.meta_info_configs = meta_info_configs or {}
        self.meta_info = meta_info
        self.graph_parser_configs = graph_parser_configs or {}
        parser = parser or ParserWrapper(data_parser=data_parser, graph_parser=graph_parser)
        self.parser = parser
        self.module = module
        self.check_point = check_point
        self.mask_prob = mask_prob
        self.mask_loss_weight=mask_loss_weight
        self.return_details = return_details
        self.return_embedding = return_embedding

    @property
    def return_embedding(self):
        return getattr(self, '_return_embedding', False)

    @return_embedding.setter
    def return_embedding(self, x):
        self._return_embedding = x

    @property
    def return_details(self):
        return getattr(self, '_details', False)

    @return_details.setter
    def return_details(self, x):
        self._details = x

    @property
    def mask_prob(self):
        return getattr(self, '_mask_prob', 0)

    @mask_prob.setter
    def mask_prob(self, x):
        self._mask_prob = x

    @property
    def mask_loss_weight(self):
        return getattr(self, '_mask_loss_weight', 1)

    @mask_loss_weight.setter
    def mask_loss_weight(self, x):
        self._mask_loss_weight = x

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

    def mask_forward(self, inputs: Graph):
        inputs = self.module.transform_graph(inputs)
        
        mask = (inputs.strings != self.graph_parser.default_code) * (torch.rand(inputs.num_nodes, device=inputs.device) < self.mask_prob)
        mask_label = inputs.strings[mask]
        inputs.strings[mask] = self.graph_parser.mask_code
        
        details = self.module.forward_internal(inputs)
        
        mask_feats = details.ndata['feats'].view(inputs.num_nodes, -1)[mask]
        mask_token_emb = self.module.str_emb(torch.arange(self.graph_parser.num_tokens, device=self.device))
        mask_pred = self.module.head.forward(mask_feats, mask_token_emb)
        return details, mask_pred, mask_label, mask

    def forward(self, inputs):
        if isinstance(inputs, pd.DataFrame):
            inputs = self.parser(inputs)
        if isinstance(inputs, dict):
            return {k: self.forward(v) for k, v in inputs.items()}
        if self.mask_prob > 0:
            details, mask_pred, mask_label, _ = self.mask_forward(inputs)
            output = details.output
            output[MASK_LOSS] = cross_entropy(mask_pred, mask_label)
            with torch.no_grad():
                output[MASK_ACCURACY] = ((mask_pred.argmax(dim=1) == mask_label) * 1.0).mean()
        else:
            details = self.module(inputs)
        if self.return_details:
            return details
        elif self.return_embedding:
            return details.readout_feats
        else:
            return details.output

    def build_model(self, data, chunksize, processes=0, analyse_frac=None, **kwargs):
        assert self.data_parser is not None, 'please set data_parser first'
        if self.graph_parser is None:
            print('graph_parser not set, start building')
            if self.meta_info is None:
                print('meta_info not set, start building')
                paths = data['path'].unique()
                if analyse_frac is not None:
                    paths = np.random.choice(paths, int(np.ceil(analyse_frac*paths.shape[0])), replace=False)
                self.meta_info = self.data_parser.analyse(
                    paths, processes=processes, configs=self.meta_info_configs, chunksize=chunksize, **kwargs)
            self.graph_parser = GraphParser.from_data(self.meta_info, **self.graph_parser_configs)
        self.build_framework()

    def build_framework(self):
        raise NotImplementedError

    def fit(self,
            train_data,
            epoches=0,
            batch_size=None,
            loss=None,
            chunksize=None,
            valid_data: list = None,
            num_workers=0,
            processes=None,
            optimizer=None,
            metrics: dict = None,
            early_stop=None,
            analyse_frac=None,
            pin_memory=False,
            max_history=0,
            device=None,
            data_process_func=None,
            **kwargs
            ):
        """

        Args:
            train_data: dataframe, 长度与样本数量相同, 包含一列path代表每个样本的文件地址
            loss:
            epoches:
            batch_size:
            valid_data:
            num_workers: dataloader的多进程数量
            processes: 分析数据时多进程的数量, 如果None, num_workers
            optimizer:
            metrics:
            early_stop:
            analyse_frac: 分析阶段的数据采样率
            pin_memory:
            max_history: 保存epoch预期结果的轮数
            **kwargs:

        Returns:

        """
        chunksize = chunksize or batch_size
        processes = processes or num_workers
        if self.mask_prob > 0:
            loss = MaskLossWrapper(loss, weight=self.mask_loss_weight)
            if metrics is None:
                metrics = {}
            metrics[MASK_LOSS] = get_mask_loss
            metrics[MASK_ACCURACY] = get_mask_accuracy

        # dataloader
        if not isinstance(train_data, DataLoader):
            train_data = CSVDataset(train_data, self.parser, func=data_process_func, **kwargs).dataloader(
                batch_size=batch_size, num_workers=num_workers, shuffle=True, pin_memory=pin_memory)

        if valid_data is not None:
            old_valid_data = valid_data
            valid_data = []
            for data in old_valid_data:
                if not isinstance(data, DataLoader):
                    data = CSVDataset(data, self.parser, func=data_process_func, **kwargs).dataloader(
                        batch_size=batch_size, num_workers=num_workers, shuffle=True, pin_memory=pin_memory)
                valid_data.append(data)

        # build module
        if self.module is None:
            self.build_model(
                data=train_data,
                chunksize=chunksize,
                processes=processes,
                analyse_frac=analyse_frac,
                **kwargs
            )
        assert not isinstance(self.data_parser, dict), 'fitting not support multi data parsers'
        assert not isinstance(self.graph_parser, dict), 'fitting not support multi graph parsers'

        if device is not None:
            self.to(device)

        # train
        trainer = Trainer(
            module=self,
            optimizer=optimizer,
            loss=loss,
            data=train_data,
            valid_data=valid_data,
            metrics=metrics,
            check_point=self.check_point,
            early_stop=early_stop,
            max_history=max_history
        )
        if epoches > 0:
            print(trainer)
            trainer.fit(epoches)
        return trainer

    def predict(self, src, **kwargs):
        mask_prob, self.mask_prob = self.mask_prob, 0
        output = predict_csv(self.parser, self, src, **kwargs)
        self.mask_prob = mask_prob
        return output

    def extra_repr(self) -> str:
        output = [
            super().extra_repr(),
            f'meta_info={self.meta_info}',
            f'parser={self.parser}',
            f'mask_prob={self.mask_prob}',
            f'mask_loss_weight={self.mask_loss_weight}',
            f'check_point="{self.check_point}"'
        ]
        return '\n'.join(output)
