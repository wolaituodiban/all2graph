import os
from typing import Iterable, Union, List, Dict

import torch
import torchaudio
from torchaudio.functional import resample
from torchaudio.transforms import Resample
from torch.utils.data import Dataset


def dir_traversal(path: str, outputs: List[str]):
    if os.path.isdir(path):
        for file_path in os.listdir(path):
            file_path = os.path.join(path, file_path)
            dir_traversal(file_path, outputs)
    else:
        outputs.append(path)


class AudioDataset(Dataset):
    def __init__(self, paths: Union[str, Iterable[str]], resample_rate: int, resample_config: dict = None,
                 cache_resampler=False, transformer: torch.nn.Module = None, **kwargs):
        """

        :param.py paths: 包含样本文件的路径，如果是文件夹，那么会遍历文件下所有文件，寻找能被读取的音频文件
        :param.py resample_rate: 重采样率
        :param.py resample_config: dict，重采样参数
        :param.py cache_resampler: 是否缓存重采样器，缓存能加速
        :param.py transformer: 转换器，如频谱转换器
        :param.py item_transform: 是否在get item阶段调用transformer
        :param.py kwargs: 加载音频的其他参数，详见torchaudio.load
        """
        if isinstance(paths, str):
            # 遍历文件夹下所有文件路径
            temp = []
            dir_traversal(paths, temp)
            paths = temp

        # 判断文件列表是否全是音频文件
        self.paths = []
        for path in Progress(paths):
            try:
                torchaudio.load(path, **kwargs)
                self.paths.append(path)
            except RuntimeError:
                pass
        print('{} files scaned, {} files added'.format(len(paths), len(self.paths)))

        # 测试config是否正确
        self.resample_rate = resample_rate
        resample_config = resample_config or {}
        if resample_config is not None:
            Resample(44100, new_freq=self.resample_rate, **resample_config)
        self.resample_config = resample_config

        # 对resampler进行cache
        if cache_resampler:
            self.resamplers: Dict[str, Resample] = {}
        else:
            self.resamplers = None

        self.transformer = transformer
        self.kwargs = kwargs

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, item) -> torch.Tensor:
        waveform, sample_rate = torchaudio.load(self.paths[item], **self.kwargs)
        if sample_rate != self.resample_rate:
            if self.resamplers is None:
                waveform = resample(waveform, sample_rate, self.resample_rate, **self.resample_config)
            else:
                if sample_rate not in self.resamplers:
                    self.resamplers[sample_rate] = Resample(sample_rate, self.resample_rate, **self.resample_config)
                waveform = self.resamplers[sample_rate](waveform)
        return waveform

    def collate_fn(self, items: List[torch.Tensor]) -> Union[torch.Tensor, (torch.Tensor, torch.Tensor)]:
        max_channels = 0
        max_frames = 0
        for item in items:
            max_channels = max(item.shape[0], max_channels)
            max_frames = max(item.shape[1], max_frames)

        output = torch.zeros(len(items), max_channels, max_frames, dtype=items[0].dtype)
        for i, item in enumerate(items):
            output[i, :item.shape[0], :item.shape[1]] = item
        if self.transformer is not None:
            return output, self.transformer(output)
        else:
            return output


class AudioMixtureDataset(AudioDataset):
    def collate_fn(self, items: List[torch.Tensor]) -> torch.Tensor:
        pass
