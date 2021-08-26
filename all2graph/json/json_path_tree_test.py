import json
from datetime import datetime

import pandas as pd

from all2graph.json import JsonPathTree, Split, TimeProcessor, Delete, Lower
from all2graph.utils import progress_wrapper


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


def preprocessor(df):
    for row in df[['crt_dte', 'json']].itertuples():
        crt_dte = datetime.strptime(row[1], '%Y-%m-%d')
        obj = json.loads(row[2])['SMALL_LOAN']
        for item in obj:
            if 'crt_tim' in item:
                crt_tim = datetime.strptime(item['crt_tim'], '%Y-%m-%d %H:%M:%S')
                crt_tim_diff = crt_dte - crt_tim
                item['crt_tim_diff_day'] = crt_tim_diff.days + int(crt_tim_diff.seconds > 0)
                item['crt_tim_day'] = crt_tim.day
                item['crt_tim_hour'] = crt_tim.hour
                item['crt_tim_weekday'] = crt_tim.weekday()
                del item['crt_tim']
            else:
                rep_tim = datetime.strptime(item['rep_tim'], '%Y-%m-%d %H:%M:%S')
                rep_tim_diff = crt_dte - rep_tim
                item['rep_tim_diff_day'] = rep_tim_diff.days + int(rep_tim_diff.seconds > 0)
                item['rep_tim_day'] = rep_tim.day
                item['rep_tim_hour'] = rep_tim.hour
                item['rep_tim_weekday'] = rep_tim.weekday()

                rep_dte = datetime.strptime(item['rep_dte'], '%Y-%m-%d')
                rep_dte_diff = crt_dte - rep_dte
                item['rep_dte_diff_day'] = rep_dte_diff.days + int(rep_dte_diff.seconds > 0)
                item['rep_dte_day'] = rep_dte.day
                item['rep_dte_weekday'] = rep_dte.weekday()
                del item['rep_tim'], item['rep_dte']
            del item['prc_amt'], item['adt_lmt'], item['avb_lmt']

            item['bsy_typ'] = [s.lower() for s in item['bsy_typ'].split('_')]
            if 'ded_typ' in item:
                item['ded_typ'] = [s.lower() for s in item['ded_typ'].split('_')]
        yield obj


def test():
    json_path_tree = JsonPathTree(
        'json',
        sample_time_col='crt_dte',
        sample_time_format='%Y-%m-%d',
        processors=[
            ('$.SMALL_LOAN',),
            ('$.*', TimeProcessor('crt_tim', '%Y-%m-%d %H:%M:%S', ['day', 'hour', 'weekday'])),
            ('$.*', TimeProcessor('rep_tim', '%Y-%m-%d %H:%M:%S', ['day', 'hour', 'weekday'])),
            ('$.*', TimeProcessor('rep_dte', '%Y-%m-%d', ['day', 'weekday'])),
            ('$.*.bsy_typ', Lower()),
            ('$.*.ded_typ', Lower()),
            ('$.*.bsy_typ', Split('_')),
            ('$.*.ded_typ', Split('_')),
            ('$.*', Delete(['crt_tim', 'rep_tim', 'rep_dte', 'prc_amt', 'adt_lmt', 'avb_lmt']))
        ]
    )
    print(json_path_tree)
    standard_answer = list(preprocessor(data))
    print(json.dumps(standard_answer, indent=2))
    assert list(json_path_tree(data)) == standard_answer, list(json_path_tree(data))



def speed():
    df = pd.concat([data] * 10000)
    json_path_tree = JsonPathTree(
        'json',
        sample_time_col='crt_dte',
        sample_time_format='%Y-%m-%d',
        processors=[
            ('$.SMALL_LOAN',),
            ('$.*', TimeProcessor('crt_tim', '%Y-%m-%d %H:%M:%S', ['day', 'hour', 'weekday'])),
            ('$.*', TimeProcessor('rep_tim', '%Y-%m-%d %H:%M:%S', ['day', 'hour', 'weekday'])),
            ('$.*', TimeProcessor('rep_dte', '%Y-%m-%d', ['day', 'weekday'])),
            ('$.*.bsy_typ', Lower()),
            ('$.*.ded_typ', Lower()),
            ('$.*.bsy_typ', Split('_')),
            ('$.*.ded_typ', Split('_')),
            ('$.*', Delete(['crt_tim', 'rep_tim', 'rep_dte', 'prc_amt', 'adt_lmt', 'avb_lmt']))
        ]
    )
    print(json_path_tree)
    for _ in progress_wrapper(json_path_tree(df), size=df.shape[0]):
        pass

    for _ in progress_wrapper(preprocessor(df), size=df.shape[0]):
        pass


if __name__ == '__main__':
    test()
    speed()
