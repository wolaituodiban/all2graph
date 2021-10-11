import os
import pandas as pd

from all2graph import MetaInfo
from all2graph import JsonParser, Timer, JiebaTokenizer
from all2graph.parsers.graph import RawGraphParser


path = os.path.dirname(__file__)
path = os.path.dirname(path)
path = os.path.dirname(path)

csv_path = os.path.join(path, 'test_data', 'MensShoePrices.csv')
df = pd.read_csv(csv_path, nrows=64)

parser = JsonParser(
    'json', flatten_dict=True, local_id_keys={'name'}, segment_value=True, self_loop=True,
    list_inner_degree=1, tokenizer=JiebaTokenizer()
)
raw_graph, global_index_mapper, local_index_mappers = parser.parse(df, progress_bar=True)

index_ids = list(global_index_mapper.values())
for mapper in local_index_mappers:
    index_ids += list(mapper.values())
meta_info = MetaInfo.from_data(raw_graph, index_nodes=index_ids, progress_bar=True)


def test_init():
    RawGraphParser.from_data(meta_info, min_df=0.01, max_df=0.95, top_k=100, top_method='max_tfidf')


def test():
    trans1 = RawGraphParser.from_data(meta_info)
    with Timer('speed'):
        graph = trans1.parse(raw_graph)
    print(graph)


if __name__ == '__main__':
    test_init()
    test()
