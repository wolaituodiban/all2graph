import os
from setuptools import setup, find_packages

# 根据是否有cython来确定ext_modules
try:
    from Cython.Build import cythonize
    ext_modules = cythonize(
        [
            'all2graph/meta_struct.py',
            'all2graph/graph/raw_graph.py',
            'all2graph/parsers/data_parser.py',
            'all2graph/parsers/json_parser.py',
            'all2graph/parsers/graph_parser.py',
            'all2graph/parsers/parser_wrapper.py',  
        ],
        exclude=[
            'all2graph/parsers/*test*.py',
        ],
        compiler_directives={
            'language_level': 3,
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


with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'README.md'), 'r') as f:
    long_description = f.read()


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
        'tqdm'
    ],
    url='https://github.com/wolaituodiban/all2graph.git',
    long_description=long_description,
    ext_modules=ext_modules,
    license='MIT',
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
)
