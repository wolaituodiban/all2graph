# 解决在服务器上运行时，dgl写config文件时，没有权限的问题
import platform
system = platform.system()
if 'linux' in system.lower():
    import os
    import sys
    old_home = os.environ['HOME']
    new_home = os.getcwd()
    os.environ['HOME'] = new_home
    try:
        import dgl
        config_path = os.path.join(new_home, '.dgl', 'config.json')
        print('all2graph has moved dgl config.json to {}'.format(config_path), file=sys.stderr)
    except ImportError:
        pass
    os.environ['HOME'] = old_home

try:
    import torch
except ImportError:
    torch = None
if torch is not None:
    from . import nn
    from . import data
else:
    print('failed to import module nn and data, no torch installed', file=sys.stderr)


from . import graph
from . import json
from . import preserves
from .parsers import DataParser, RawGraphParser, ParserWrapper
from .factory import Factory
from .meta import MetaInfo, MetaValue, MetaNumber, MetaString
from .stats import ECDF, Discrete, Distribution
from .utils import *
from .globals import *
from .meta_struct import MetaStruct
from .version import __version__
