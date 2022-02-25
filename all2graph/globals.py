import json


EPSILON = 2e-6
NULL = list(json.loads(json.dumps({None: None})).keys())[0]
TRUE = json.dumps([True])[1:-1]
FALSE = json.dumps([False])[1:-1]
SID = 'sid'
KEY = 'key'
VALUE = 'value'
TARGET = 'target'
EDGE = 'edge'
READOUT = 'readout'

KEY2KEY = (KEY, EDGE, KEY)
KEY2VALUE = (KEY, EDGE, VALUE)
KEY2TARGET = (KEY, EDGE, TARGET)
VALUE2VALUE = (VALUE, EDGE, VALUE)
VALUE2TARGET = (VALUE, EDGE, TARGET)

del json
