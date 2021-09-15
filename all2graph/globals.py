import json

EPSILON = 2e-6
NULL = list(json.loads(json.dumps({None: None})).keys())[0]
TRUE = json.dumps([True])[1:-1]
FALSE = json.dumps([False])[1:-1]

del json

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

PRESERVED_WORDS = [v for k, v in locals().items() if k[:2] != '__' and isinstance(v, str)]

COMPONENT_ID = ' '.join([COMPONENT, ID])
META_NODE_ID = ' '.join([META, NODE, ID])
META_EDGE_ID = ' '.join([META, EDGE, ID])

SRC_KEY_WEIGHT = ' '.join([SRC, KEY, WEIGHT])
SRC_KEY_BIAS = ' '.join([SRC, KEY, BIAS])

DST_KEY_BIAS = ' '.join([DST, KEY, BIAS])
DST_KEY_WEIGHT = ' '.join([DST, KEY, WEIGHT])

SRC_VALUE_WEIGHT = ' '.join([SRC, VALUE, WEIGHT])
SRC_VALUE_BIAS = ' '.join([SRC, VALUE, BIAS])

DST_VALUE_WEIGHT = ' '.join([DST, VALUE, WEIGHT])
DST_VALUE_BIAS = ' '.join([DST, VALUE, BIAS])

NODE_WEIGHT = ' '.join([NODE, WEIGHT])
NODE_BIAS = ' '.join([NODE, BIAS])
