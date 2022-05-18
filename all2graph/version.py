import subprocess
import os
from .globals import __version__

old_dir = os.getcwd()
try:
    os.chdir(os.path.dirname(os.path.dirname(__file__)))
    __version__ = '.'.join([__version__, subprocess.check_output(['git', 'rev-parse', 'HEAD']).decode('utf-8').strip()])
except:
    pass
finally:
    os.chdir(old_dir)
