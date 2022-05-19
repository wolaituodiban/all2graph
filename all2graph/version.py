import subprocess
import os
from .globals import __version__

old_dir = os.getcwd()
try:
    old_home = os.environ['HOME']
    os.environ['HOME'] = os.path.dirname(os.path.dirname(__file__))
    __version__ = '.'.join([__version__, subprocess.check_output(['git', 'rev-parse', 'HEAD']).decode('utf-8').strip()])
except:
    pass
finally:
    os.environ['HOME'] = old_home
