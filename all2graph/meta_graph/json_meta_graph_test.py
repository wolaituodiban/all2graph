import os
import json
import pandas as pd
from all2graph.meta_graph import JsonMetaGraph


def test_json_graph():
    path = os.path.dirname(__file__)
    path = os.path.dirname(path)
    path = os.path.dirname(path)
    path = os.path.join(path, 'test_data', 'MensShoePrices.csv')
    df = pd.read_csv(path)
    df = df.iloc[:2000]
    meta_graph = JsonMetaGraph.from_data(df.dateupdated, df.json)
    print(len(json.dumps(meta_graph.to_json())))
