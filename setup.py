import os
from setuptools import setup, find_packages

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
    packages=find_packages(),
    install_requires=[
        'dgl>=0.6.0',
        'torch>=1.5.0',
        'jieba>=0.39',
        'numpy',
        'pandas',
        'scipy',
        'jsonpromax>=0.2.4'
    ],
)
