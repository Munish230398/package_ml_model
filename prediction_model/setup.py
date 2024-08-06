import io
import os 
from pathlib import Path

from setuptools import find_packages, setup 

NAME = 'prediction_model'
DESCRIPTION = ' Employee attrition prdiction model'
AUTHOR = 'Munish Singh Danu'
REQUIRES_PYTHON = '>=3.10.14'

pwd = os.path.abspath(os.path.dirname(__file__))

## get the list of packages to be installed
def list_reqs(fname='requirement.txt'):
    with io.open(os.path.join(pwd,fname), encoding='utf-8') as f:
        return f.read().splitlines()
    
try:
    with io.open(os.path.join(pwd, 'README.md'), encoding='utf-8') as f:
        long_description = '\n' + f.read()
except FileNotFoundError:
    long_description = DESCRIPTION


## load the packages version
ROOT_DIR = Path(__file__).resolve().parent
PACKAGE_DIR = ROOT_DIR 
about = {}
with open(PACKAGE_DIR/'VERSION') as f:
    _version = f.read().strip()
    about['__version__'] = _version



setup(
    name = NAME,
    version = about['__version__'],
    description = DESCRIPTION,
    long_description = long_description,
    long_description_content_type = 'text/markdown',
    author = AUTHOR,
    python_requires = REQUIRES_PYTHON,
    packages = find_packages(exclude=['tests']),
    package_data = {'prediction_model': ['VERSION']},
    install_requires = list_reqs(),
    extras_require = {},
    include_package_data  = True,


)