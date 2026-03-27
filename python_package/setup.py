
from setuptools import find_packages
from setuptools import setup
import setuptools
from distutils.command.build import build as _build
import subprocess

REQUIRED_PACKAGES = [
    'sacrebleu',
    'tqdm',
    'docopt',
    'sentencepiece',
    'cloudml-hypertune',
    'nltk'
]

setup(
    name='trainer',
    version='0.1',
    install_requires=REQUIRED_PACKAGES,
    packages=find_packages(),
    py_modules=['run', 'download_data', 'vocab'],
    include_package_data=True,
    description='Vertex AI | Training | PyTorch | Neural Machine Translation with RNN | Python Package'
)
