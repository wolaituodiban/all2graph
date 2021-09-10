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


PRESERVED_WORDS = [i for i in globals() if isinstance(i, str)]

COMPONENT_ID = ' '.join([COMPONENT, ID])
META_NODE_ID = ' '.join([META, NODE, ID])
META_EDGE_ID = ' '.join([META, EDGE, ID])

SRC_KEY_WEIGHT = ' '.join([SRC, KEY, WEIGHT])
DST_KEY_WEIGHT = ' '.join([DST, KEY, WEIGHT])
EDGE_KEY_WEIGHT = ' '.join([EDGE, KEY, WEIGHT])

SRC_KEY_BIAS = ' '.join([SRC, KEY, BIAS])
DST_KEY_BIAS = ' '.join([DST, KEY, BIAS])
EDGE_KEY_BIAS = ' '.join([EDGE, KEY, BIAS])

SRC_VALUE_WEIGHT = ' '.join([SRC, VALUE, WEIGHT])
DST_VALUE_WEIGHT = ' '.join([DST, VALUE, WEIGHT])
EDGE_VALUE_WEIGHT = ' '.join([EDGE, VALUE, WEIGHT])

SRC_VALUE_BIAS = ' '.join([SRC, VALUE, BIAS])
DST_VALUE_BIAS = ' '.join([DST, VALUE, BIAS])
EDGE_VALUE_BIAS = ' '.join([EDGE, VALUE, BIAS])

NODE_KEY_WEIGHT_1 = ' '.join([NODE, KEY, WEIGHT, '1'])
NODE_KEY_BIAS_1 = ' '.join([NODE, KEY, BIAS, '1'])
NODE_KEY_WEIGHT_2 = ' '.join([NODE, KEY, WEIGHT, '2'])
NODE_KEY_BIAS_2 = ' '.join([NODE, KEY, BIAS, '2'])

NODE_VALUE_WEIGHT_1 = ' '.join([NODE, VALUE, WEIGHT, '1'])
NODE_VALUE_BIAS_1 = ' '.join([NODE, VALUE, BIAS, '1'])
NODE_VALUE_WEIGHT_2 = ' '.join([NODE, VALUE, WEIGHT, '2'])
NODE_VALUE_BIAS_2 = ' '.join([NODE, VALUE, BIAS, '2'])
