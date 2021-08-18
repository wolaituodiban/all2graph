import json

EPSILON = 2e-6
SECOND_DIFF = 'second_diff'
NULL = list(json.loads(json.dumps({None: None})).keys())[0]
TRUE = json.dumps([True])[1:-1]
FALSE = json.dumps([False])[1:-1]
