import subprocess
import os
from .globals import __version__


try:
    old_dir = os.getcwd()
    os.chdir(os.path.dirname(os.path.dirname(__file__)))
    __version__ = '.'.join([__version__, subprocess.check_output(['git', 'rev-parse', 'HEAD']).decode('utf-8').strip()])
    os.chdir(old_dir)
except:
    pass
