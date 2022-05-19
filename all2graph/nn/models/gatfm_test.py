import os
import platform

if 'darwin' in platform.system().lower():
    os.environ['OMP_NUM_THREADS'] = '1'
