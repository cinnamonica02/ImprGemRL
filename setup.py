# setup.py
from setuptools import setup, find_packages

setup(
    name="ImprGemRL",
    version="0.1",
    packages=find_packages(),
    install_requires=[
        'litellm',
        'datasets',
        'python-dotenv'
    ],
)