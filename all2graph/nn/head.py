from typing import Dict

import torch

from .feedforward import FeedForward


class Head(FeedForward):
    def __init__(self, in_feats, out_feats=1, **kwargs):
        # 如果使用norm，会让target feats的差异消失，因此不能用norm
        super().__init__(in_feats, out_feats=out_feats, norm=torch.nn.Identity(), residual=False, **kwargs)

    def forward(self, readout_feats: torch.Tensor,
                target_feats: Dict[str, torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        output = {}
        for target, feats in target_feats.items():
            feats = feats.expand(readout_feats.shape[0], feats.shape[0])
            feats = torch.cat([readout_feats, feats], dim=1)
            pred = super().forward(feats).squeeze(-1)
            output[target] = pred
        return output
