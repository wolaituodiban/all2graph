from typing import Dict

import torch

from .feedforward import FeedForward


class Head(FeedForward):
    def __init__(self, in_feats, out_feats=1, dropout=0, activation='prelu', norm_first=True):
        super().__init__(2 * in_feats, middle_feats=in_feats, out_feats=out_feats, dropout=dropout,
                         activation=activation, norm_first=norm_first, residual=False)

    def forward(self, readout_feats: torch.Tensor,
                target_feats: Dict[str, torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        output = {}
        for target, feats in target_feats.items():
            feats = feats.expand(readout_feats.shape[0], feats.shape[0])
            feats = torch.cat([readout_feats, feats], dim=1)
            pred = super().forward(feats).squeeze(-1)
            output[target] = pred
        return output
