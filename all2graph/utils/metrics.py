import numpy as np
from inspect import isfunction


def ks_score(y_true, y_score):
    from sklearn.metrics import roc_curve
    fpr, tpr, _ = roc_curve(y_true, y_score)
    return max(tpr - fpr)


class Metric:
    """
    a wrapper for func, which will make the func which is label first
    Examples:
        >>> from sklearn.metrics import roc_auc_score
        ... y_hat = ...
        ... y = ...
        ... auc = Metric(roc_auc_score, label_first=True)
        ... auc(y_hat, y)
        0.5

        >>> y_hat = {'a': ..., 'b': ...}
        ... y = {'a': ..., 'b': ...}
        ... auc = Metric(roc_auc_score, label_first=True)
        ... auc(y_hat, y)
        {'a': 0.5, 'b': 0.5}

        >>> y_hat = [..., ...]
        ... y = [..., ...]
        ... auc = Metric(roc_auc_score, label_first=True)
        ... auc(y_hat, y)
        [0.5, 0.5]

    Args:
        func: metric func, list of funcs or dict of funcs
        label_first: whether the input metric func receive label as the first argument

    Returns:
        new function
    """
    def __init__(self, func, label_first):
        self.func = func
        self.label_first = label_first

    def __repr__(self):
        if isfunction(self.func):
            func_repr = self.func.__name__
        else:
            func_repr = self.func
        return '{}(func={}, label_first={})'.format(self.__class__.__name__, func_repr, self.label_first)

    def call(self, label, pred):
        if self.label_first:
            return self.func(label, pred)
        else:
            return self.func(pred, label)

    def __call__(self, label, pred):
        if isinstance(pred, np.ndarray):
            mask = np.bitwise_not(np.isnan(label))
            return self.call(pred=pred[mask], label=label[mask])
        elif isinstance(pred, list):
            return [self(pred=_p, label=_l) for _p, _l in zip(pred, label)]
        elif isinstance(pred, dict):
            return {key: self(pred=pred[key], label=label[key]) for key in set(pred).intersection(label)}
        raise TypeError('only accept "numpy", "list", "dict"')
