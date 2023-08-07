# 解决在服务器上运行时，dgl写config文件时，没有权限的问题
import os
import platform
import sys
from .version import __version__
print('all2graph: version={}'.format(__version__), file=sys.stderr)
if 'linux' in platform.system().lower():
    old_home = os.environ['HOME']
    new_home = os.getcwd()
    os.environ['HOME'] = new_home
    try:
        import dgl
        config_path = os.path.join(new_home, '.dgl', 'config.json')
        print('all2graph: move ~/.dgl/config.json to {}'.format(config_path), file=sys.stderr)
    except ImportError:
        pass
    finally:
        os.environ['HOME'] = old_home

from . import nn
from . import data
from .graph import EventGraph, EventSet, Event, EventGraphV2, EventGraphV3
from .parser import Parser, ParserV2, ParserV3
from .utils import *
from .globals import *
