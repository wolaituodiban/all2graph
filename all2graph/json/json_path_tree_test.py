import json
import pandas as pd

from all2graph.json import JsonPathTree, GetAttr, SplitString, TimeProcessor, Delete, Lower
from all2graph.json.json_node_processors import JsonNodeProcessor


def test_parsing():
    json_path_tree = JsonPathTree(
        'json',
        processors=[
            ("$.firstname", JsonNodeProcessor()),
            ("$['firstname']", JsonNodeProcessor()),
            ("$..['type']", JsonNodeProcessor()),
            ("$[*]", JsonNodeProcessor()),
            ("$.*", JsonNodeProcessor()),
            ("$['phoneNumber']..['type']", JsonNodeProcessor()),
            ("$['phoneNumber'][0].number", JsonNodeProcessor()),
            ("$.*", JsonNodeProcessor()),
        ]
    )
    print(json_path_tree)


def test():
    data = {
        'SMALL_LOAN': [
            {
                'ord_no': 'CH202007281033864',
                'bsy_typ': 'CASH',
                'prc_amt': 3600.0,
                'crt_tim': '2020-07-28 16:54:31',
                'adt_lmt': 3600.0,
                'avb_lmt': 0.0,
                'avb_lmt_rat': 0.0
            },
            {
                'ord_no': 'CH202007281033864',
                'bsy_typ': 'CASH',
                'stg_no': '1',
                'rep_dte': '2020-08-28',
                'rep_tim': '2020-08-28 08:35:05',
                'prc_amt': -286.93,
                'ded_typ': 'AUTO_DEDUCT',
                'adt_lmt': 3600.0,
                'avb_lmt': 286.93,
                'avb_lmt_rat': 0.079703
            },
            {
                'ord_no': 'CH202007281033864',
                'bsy_typ': 'CASH',
                'stg_no': '2',
                'rep_dte': '2020-09-28',
                'rep_tim': '2020-09-28 10:17:18',
                'prc_amt': -289.15,
                'ded_typ': 'MANUAL_REPAY',
                'adt_lmt': 3600.0,
                'avb_lmt': 576.08,
                'avb_lmt_rat': 0.160022
            }
        ]
    }
    data = pd.DataFrame(
        {
            'json': [json.dumps(data)],
            'crt_dte': '2020-10-09'
        }
    )

    json_path_tree = JsonPathTree(
        'json',
        sample_time_col='crt_dte',
        sample_time_format='%Y-%m-%d',
        processors=[
            ('$', GetAttr('SMALL_LOAN')),
            ('$.*', TimeProcessor('crt_tim', '%Y-%m-%d %H:%M:%S', ['day', 'hour', 'weekday'])),
            ('$.*', TimeProcessor('rep_tim', '%Y-%m-%d %H:%M:%S', ['day', 'hour', 'weekday'])),
            ('$.*', TimeProcessor('rep_dte', '%Y-%m-%d', ['day', 'weekday'])),
            ('$.*.bsy_typ', Lower()),
            ('$.*.ded_typ', Lower()),
            ('$.*.bsy_typ', SplitString('_')),
            ('$.*.ded_typ', SplitString('_')),
            ('$.*', Delete(['crt_tim', 'rep_tim', 'rep_dte', 'prc_amt', 'adt_lmt', 'avb_lmt']))
        ]
    )
    assert list(json_path_tree(data)) == [
        [
            {
                'ord_no': 'CH202007281033864',
                'bsy_typ': ['cash'],
                'avb_lmt_rat': 0.0,
                'crt_tim_diff_day': 73,
                'crt_tim_day': 28,
                'crt_tim_hour': 16,
                'crt_tim_weekday': 1
            },
            {
                'ord_no': 'CH202007281033864',
                'bsy_typ': ['cash'],
                'stg_no': '1',
                'ded_typ': ['auto', 'deduct'],
                'avb_lmt_rat': 0.079703,
                'rep_tim_diff_day': 42,
                'rep_tim_day': 28,
                'rep_tim_hour': 8,
                'rep_tim_weekday': 4,
                'rep_dte_diff_day': 42,
                'rep_dte_day': 28,
                'rep_dte_weekday': 4
            },
            {
                'ord_no': 'CH202007281033864',
                'bsy_typ': ['cash'],
                'stg_no': '2',
                'ded_typ': ['manual', 'repay'],
                'avb_lmt_rat': 0.160022,
                'rep_tim_diff_day': 11,
                'rep_tim_day': 28,
                'rep_tim_hour': 10,
                'rep_tim_weekday': 0,
                'rep_dte_diff_day': 11,
                'rep_dte_day': 28,
                'rep_dte_weekday': 0
            }
        ]
    ], list(json_path_tree(data))
    print(json_path_tree)


if __name__ == '__main__':
    test_parsing()
    test()
