import json

EPSILON = 2e-6
NULL = list(json.loads(json.dumps({None: None})).keys())[0]
TRUE = json.dumps([True])[1:-1]
FALSE = json.dumps([False])[1:-1]

JSON_KEY_WORDS = [i for i in globals() if isinstance(i, str)]

del json

ID = 'id'
META = 'meta'
COMPONENT = 'component'
NODE = 'node'
EDGE = 'edge'
NAME = 'name'
NUMBER = 'number'
VALUE = 'value'
KEY = 'key'
QUERY = 'query'
WEIGHT = 'weight'
BIAS = 'bias'
ATTENTION = 'attention'
FEED = 'feed'
FORWARD = 'forward'
FEATURE = 'feature'

PRESERVED_WORDS = [i for i in globals() if isinstance(i, str)]

COMPONENT_ID = ' '.join([COMPONENT, ID])
META_NODE_ID = ' '.join([META, NODE, ID])
META_EDGE_ID = ' '.join([META, EDGE, ID])
ATTENTION_WEIGHT = ' '.join([ATTENTION, WEIGHT])
ATTENTION_KEY_WEIGHT = ' '.join([ATTENTION, KEY, WEIGHT])
ATTENTION_KEY_BIAS = ' '.join([ATTENTION, KEY, BIAS])
ATTENTION_VALUE_WEIGHT = ' '.join([ATTENTION, VALUE, WEIGHT])
ATTENTION_VALUE_BIAS = ' '.join([ATTENTION, VALUE, BIAS])
FEED_FORWARD_WEIGHT = ' '.join([FEED, FORWARD, WEIGHT])
FEED_FORWARD_BIAS = ' '.join([FEED, FORWARD, BIAS])
