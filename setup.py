import os
from setuptools import setup, find_packages
# from Cython.Build import cythonize


VERSION_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'all2graph/globals.py')


def get_version():
    ns = {}
    with open(VERSION_FILE) as f:
        exec(f.read(), ns)
    return ns['__version__']


setup(
    name='all2graph',
    version=get_version(),
    author='xiaotian chen',
    author_email='wolaituodiban@gmail.com',
    packages=find_packages(),
    install_requires=[
        # 'dgl>=0.6.0',
        # 'torch>=1.5.0',
        'numpy',
        'pandas',
        'scipy',
    ],
    # ext_modules=cythonize([
    #     'all2graph/meta_struct.py',
    #     'all2graph/graph/raw_graph.py',
    #     'all2graph/parsers/json_parser.py',
    # ])
)
