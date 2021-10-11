import json


NULL = list(json.loads(json.dumps({None: None})).keys())[0]
TRUE = json.dumps([True])[1:-1]
FALSE = json.dumps([False])[1:-1]
NODE = 'node'
SRC = 'src'
DST = 'dst'
KEY = 'key'
META = 'meta'
WEIGHT = 'weight'
BIAS = 'bias'
QUERY = 'query'
VALUE = 'value'
NUMBER = 'number'
TARGET = 'target'
READOUT = 'readout'


PRESERVED_WORDS = [v for k, v in locals().items() if k[:2] != '__' and isinstance(v, str)]
