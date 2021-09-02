import json

EPSILON = 2e-6
NULL = list(json.loads(json.dumps({None: None})).keys())[0]
TRUE = json.dumps([True])[1:-1]
FALSE = json.dumps([False])[1:-1]
META = 'meta'
PRESERVED_WORDS = [NULL, TRUE, FALSE, META]

COMPONENT_IDS = 'component_ids'
META_NODE_IDS = 'meta_node_ids'
META_EDGE_IDS = 'meta_edge_ids'
NAMES = 'names'
NUMBERS = 'numbers'
VALUES = 'values'

del json
