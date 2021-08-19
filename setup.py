import os
from setuptools import setup, find_packages
# from Cython.Build import cythonize

VERSION_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'all2graph/version.py')


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
    packages=find_packages(include=('all2graph/',), exclude=('*test*',)),
    install_requires=[
        'toad',
        'dgl',
        'networkx',
        'pandas',
        'numpy',
        'scipy',
        'jieba'
    ],
    # ext_modules=cythonize([
    #     # 'all2graph/graph/graph.py'
    #     # 'all2graph/meta_node/meta_json/meta_json_value.py',
    #     # 'all2graph/meta_node/meta_json/meta_string.py',
    #     # "all2graph/stats/ecdf.py",
    # ])
)
