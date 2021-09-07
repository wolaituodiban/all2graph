import json

EPSILON = 2e-6
NULL = list(json.loads(json.dumps({None: None})).keys())[0]
TRUE = json.dumps([True])[1:-1]
FALSE = json.dumps([False])[1:-1]
META = 'meta'
PRESERVED_WORDS = [NULL, TRUE, FALSE, META]

COMPONENT_ID = 'component_id'
META_NODE_ID = 'meta_node_id'
META_EDGE_ID = 'meta_edge_id'
NAME = 'name'
NUMBER = 'number'
VALUE = 'value'

del json
