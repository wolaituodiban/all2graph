import os
import sys
import warnings
import numpy as np
import pandas as pd

warnings.filterwarnings('ignore')


class HiddenPrints:
    def __init__(self, disable):
        self.disable = disable

    def __enter__(self):
        if self.disable:
            self._original_stdout = sys.stdout
            sys.stdout = open(os.devnull, 'w')

    def __exit__(self, exc_type, exc_val, exc_tb):
        if hasattr(self, '_original_stdout'):
            sys.stdout.close()
            sys.stdout = self._original_stdout


def feature_searching(
        features,
        eval_fn,
        num_features,
        is_higher_better,
        iter_times=1000,
        lr=1,
        init_score=None,
        file_path=None,
        hidden_print=True,
        equal_prob=False
):
    if isinstance(features, list):
        weight = pd.DataFrame({0: {f: 1.0 for f in features}})
    elif isinstance(features, dict):
        weight = pd.DataFrame({0: features})
    else:
        raise TypeError('features must be one list or dict')
    weight[0] /= weight[0].sum()
    last_round_score = init_score

    for i in range(iter_times):
        if equal_prob:
            p = None
        else:
            p = weight.iloc[:, -1].values
        selected_features = np.random.choice(weight.index, size=num_features, replace=False, p=p)
        with HiddenPrints(hidden_print):
            score = eval_fn(selected_features)
        if last_round_score is None:
            last_round_score = score
        else:
            impact = (score / last_round_score - 1) * (2 * is_higher_better - 1) * lr
            weight[i] = weight[i - 1].copy(deep=True)
            weight.loc[selected_features, i] += weight.loc[selected_features, i] * impact
            weight[i] /= weight[i].sum()
            if file_path:
                weight.T.to_csv(file_path, index=False)
            print('round {}, score {}, impact {}'.format(i, score, impact))
            last_round_score = score
    return weight.T
