import json
import pandas as pd
from all2graph.json import JsonPreProcessor, TimeProcessor, parsing_json_path


def test_parsing():
    data = {
        "firstName": "John",
        "lastName": "doe",
        "age": 26,
        "address": {
            "streetAddress": "naist street",
            "city": "Nara",
            "postalCode": "630-0192"
        },
        "phoneNumbers": [
            {
                "type": "iPhone",
                "number": "0123-4567-8888"
            },
            {
                "type": "home",
                "number": "0123-4567-8910"
            }
        ]
    }
    print(parsing_json_path("$['phoneNumbers'][*]['type']", data))


def test():
    data = [
        {
            'time1': '2020-12-02',
            'data': [
                {'time2': '2020-01-06 09:49:59'},
                {'time2': '2020-03-12 18:09:23'}
            ]
        }
    ]
    data = pd.DataFrame(data)
    data['data'] = list(map(json.dumps, data.data))

    preprocessors = JsonPreProcessor(
        root_name='data',
        sample_time_col='time1',
        sample_time_format='%Y-%m-%d',
        time_processors={
            "[0]['time2']": TimeProcessor('%Y-%m-%d %H:%M:%S', ('day', 'hour'))
        }
    )
    for obj in preprocessors(data):
        print(obj)


if __name__ == '__main__':
    test_parsing()
    # test()
