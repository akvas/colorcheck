from setuptools import setup

with open('README.md', 'r') as f:
    long_description = f.read()

setup(
    name='colorcheck',
    version='0.1',
    author='Andreas Kvas',
    description='A python package to simulate color vision deficiency for scientific figures',
    install_requires=['matplotlib', 'daltonlens', 'colorspacious', 'numpy'],
    packages=['colorcheck']
)
