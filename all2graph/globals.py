import json

EPSILON = 2e-6
NULL = list(json.loads(json.dumps({None: None})).keys())[0]
TRUE = json.dumps([True])[1:-1]
FALSE = json.dumps([False])[1:-1]

del json

SEP = ' '
ID = 'id'
META = 'meta'
COMPONENT = 'component'
NODE = 'node'
EDGE = 'edge'
SRC = 'src'
DST = 'dst'
KEY = 'key'
NUMBER = 'number'
VALUE = 'value'
QUERY = 'query'
WEIGHT = 'weight'
BIAS = 'bias'
FEATURE = 'feature'
ATTENTION = 'attention'
READOUT = 'readout'
TYPE = 'type'
TARGET = 'target'

PRESERVED_WORDS = [v for k, v in locals().items() if k[:2] != '__' and isinstance(v, str)]

COMPONENT_ID = SEP.join([COMPONENT, ID])
META_NODE_ID = SEP.join([META, NODE, ID])
META_EDGE_ID = SEP.join([META, EDGE, ID])
