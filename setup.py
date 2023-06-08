import os
from setuptools import setup, find_packages


def get_version():
    ns = {}
    with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'all2graph/version.py')) as f:
        exec(f.read(), ns)
    return ns['__version__']

readme_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'README.md')
with open(readme_path, 'r', encoding='utf-8') as f:
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
        'tqdm',
        'jieba'
    ],
    url='https://github.com/wolaituodiban/all2graph.git',
    long_description=long_description,
    license='MIT',
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
)
