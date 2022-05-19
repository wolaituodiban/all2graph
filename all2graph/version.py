import platform
import os
import subprocess
from .globals import __version__


def append_version():
    return '.'.join(
        [__version__, subprocess.check_output(['git', 'rev-parse', 'HEAD']).decode('utf-8').strip()])


all2graph_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if 'linux' in platform.system().lower():
    old_home = os.environ['HOME']
    try:
        os.environ['HOME'] = all2graph_root
        __version__ = append_version()
    except:
        pass
    finally:
        os.environ['HOME'] = old_home
else:
    old_dir = os.getcwd()
    try:
        os.chdir(all2graph_root)
        __version__ = append_version()
    except:
        pass
    finally:
        os.chdir(old_dir)
