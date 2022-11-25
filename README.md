# colorcheck

colorcheck is a simple tool to simulate color vision deficiency (CVD) in images and colormaps.
Running
```
python -m colorcheck input_image.png
```
will simulate the effect of Deuteranopia, Protanopia, Tritanopia, and Monochromacy for the given image.

## Installation

The recommended way to install colorcheck is in a [conda](https://docs.conda.io/en/latest/index.html) environment:
```
conda create -n colorcheck_env
conda activate colorcheck_env
```
Then, install all dependencies:
```
conda install matplotlib pip
python -m pip install daltonlens colorspacious
```
You can also use the `environment.yml` file to directly create a new environment with all dependencies:
```
conda env create -f environment.yml
```

To install the current development version of the package, first clone the repository or download the zip archive.
In the root directory of the package (i.e. the directory containing the ``setup.py`` file), running
```
python -m pip install .
```
will install the package.
If you want to modify or extend the package, you can install it in develop mode by running
```
python -m pip install -e .
```
instead.