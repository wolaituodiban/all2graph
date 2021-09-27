from .data import *
from .graph import *
from .json import *
from .meta import *
from .stats import *
from .utils import *
from .globals import *
from .meta_struct import MetaStruct
from .version import __version__


# 解决在服务器上运行时，dgl写config文件时，没有权限的问题
import platform
system = platform.system()
if 'linux' in platform.system().lower():
    import os
    old_home = os.environ['HOME']
    new_home = os.getcwd()
    os.environ['HOME'] = new_home
    try:
        import dgl
        config_path = os.path.join(new_home, '.dgl', 'config.json')
        print('all2graph has moved dgl config.json to {}'.format(config_path))
    except ImportError:
        pass
    os.environ['HOME'] = old_home
