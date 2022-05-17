import os

import numpy as np
import pandas as pd
import torch

if __name__ == '__main__':
    dir_path = os.path.dirname(__file__)
    for filename in os.listdir(dir_path):
        if not filename.endswith('.py'):
            test_data_df = pd.read_csv(os.path.join(dir_path, filename, 'test_data.csv.zip'))
            model = torch.load(os.path.join(dir_path, filename, 'model.th'))
            pred_df = model.predict(test_data_df, processes=0, drop_data_cols=False)
            for target in model.data_parser.targets:
                assert np.allclose(test_data_df[target+'_pred'].values, pred_df[target+'_pred'].values), target
