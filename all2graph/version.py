import os
import subprocess
from .globals import __version__


try:
    old_dir = os.getcwd()
    try:
        os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        __version__ = '.'.join(
            [__version__, subprocess.check_output(['git', 'rev-parse', 'HEAD']).decode('utf-8').strip()])
    except:
        pass
    finally:
        os.chdir(old_dir)
except:
    pass
