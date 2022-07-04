from typing import Dict, Union

import torch

from .feedforward import FeedForward


class Head(FeedForward):
    def __init__(self, in_feats, out_feats=1, **kwargs):
        # 如果使用norm，会让target feats的差异消失，因此不能用norm
        super().__init__(in_feats, out_feats=out_feats, norm=torch.nn.Identity(), residual=False, **kwargs)

    def forward(self, readout_feats: torch.Tensor,
                target_feats: Union[Dict[str, torch.Tensor], torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """
        Inputs: 
            readout_feats: (N, E)
            target_feats: 如果是dict, 那么(E, ), 否则(T, E)
        Outputs:
            如果target_feats是dict, 那么(N, ), 否则(N, T)
        """
        if isinstance(target_feats, dict):
            output = {}
            for target, feats in target_feats.items():
                feats = feats.expand(readout_feats.shape[0], feats.shape[0])
                feats = torch.cat([readout_feats, feats], dim=1)
                pred = super().forward(feats).squeeze(-1)
                output[target] = pred
            return output
        else:
            readout_feats = readout_feats.unsqueeze(1).expand(readout_feats.shape[0], target_feats.shape[0], readout_feats.shape[1])
            target_feats = target_feats.unsqueeze(0).expand(readout_feats.shape[0], target_feats.shape[0], target_feats.shape[1])
            feats = torch.cat([readout_feats, target_feats], dim=2)
            return super().forward(feats).squeeze(-1)
