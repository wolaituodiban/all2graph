import os
from setuptools import setup, find_packages

# 根据是否有cython来确定ext_modules
try:
    from Cython.Build import cythonize
    ext_modules = cythonize(
        [
            'all2graph/meta_struct.py',
            'all2graph/graph/raw_graph.py',
            'all2graph/parsers/*parser*.py',
        ],
        exclude=[
            'all2graph/parsers/*test*.py',
        ],
        compiler_directives={
            'profile': False
        }
    )
except ImportError:
    ext_modules = None


def get_version():
    ns = {}
    with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'all2graph/globals.py')) as f:
        exec(f.read(), ns)
    return ns['__version__']


setup(
    name='all2graph',
    version=get_version(),
    author='xiaotian chen',
    author_email='wolaituodiban@gmail.com',
    packages=find_packages(),
    install_requires=[
        'numpy',
        'pandas',
        'scipy',
    ],
    ext_modules=ext_modules
)
