from setuptools import setup, find_packages

setup(
    name='image_classification',
    version='0.1',
    packages=find_packages(),
    install_requires=open('requirements.txt').read().splitlines(),
)
