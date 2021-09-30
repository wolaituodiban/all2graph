import torch


class BCEWithLogitsLoss:
    def __init__(self, target_weight=None, **kwargs):
        self.loss = torch.nn.BCEWithLogitsLoss(**kwargs)
        self.target_weight = target_weight

    def __call__(self, output, label):
        loss_sum = 0
        weight_sum = 0
        for k, la in label.items():
            o = output[k]
            la = la.to(o.device)
            mask = torch.bitwise_not(torch.isnan(la))
            o = o[mask]
            la = la[mask]
            if la.shape[0] == 0:
                continue
            loss = self.loss(o, la)
            if self.target_weight is None:
                loss_sum = loss_sum + loss
            else:
                loss_sum = loss_sum + loss * self.target_weight[k]
                weight_sum += self.target_weight[k]
        if weight_sum > 0:
            loss_sum = loss_sum / weight_sum
        return loss_sum
