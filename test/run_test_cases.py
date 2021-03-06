import os
import json
import numpy as np
import pandas as pd
import torch
import all2graph as ag

if __name__ == '__main__':
    dir_path = os.path.dirname(__file__)
    for filename in os.listdir(dir_path):
        data_path = os.path.join(dir_path, filename, 'test_data.csv.zip')
        if not os.path.exists(data_path):
            continue
        test_data_df = pd.read_csv(data_path)

        framework_path = os.path.join(dir_path, filename, 'framework.th')
        parser_wrapper_path = os.path.join(dir_path, filename, 'parser_wrapper.json')
        framework = torch.load(framework_path)
        with open(parser_wrapper_path, 'r') as file:
            parser_wrapper = ag.ParserWrapper.from_json(json.load(file))
        model = ag.nn.Model(parser=parser_wrapper, module=framework)
        print(model)
        pred_df = model.predict(test_data_df, processes=0, drop_data_cols=False)
        for target in model.data_parser.targets:
            assert np.allclose(test_data_df[target+'_pred'].values, pred_df[target+'_pred'].values), target

        model_path = os.path.join(dir_path, filename, 'model.th')
        if os.path.exists(model_path):
            model = torch.load(model_path)
            print(model)
            pred_df = model.predict(test_data_df, processes=0, drop_data_cols=False)
            for target in model.data_parser.targets:
                assert np.allclose(test_data_df[target+'_pred'].values, pred_df[target+'_pred'].values), target
