import os
from setuptools import setup, find_packages


# Utility function to read the README file.
# Used for the long_description.  It's nice, because now 1) we have a top level
# README file and 2) it's easier to type in the README file than to put a raw
# string in below ...
def read(fname):
    return open(os.path.join(os.path.dirname(__file__), fname)).read()


# Read metadata from version file
def get_version():
    with open("scatnet_learn/__init__.py") as f:
        for line in f:
            if line.startswith("__version__"):
                return line[15:-2]
    raise Exception("Could not find version number")


setup(
    name='scatnet_learn',
    author="Fergal Cotter",
    version=get_version(),
    author_email="fbc23@cam.ac.uk",
    description=("Wavelet based image classifier for cifar datasets"),
    license="MIT",
    keywords="wavelet, complex wavelet, DT-CWT, tensorflow, cifar, classifier",
    url="https://github.com/fbcotter/scatnet_learn",
    packages=find_packages(exclude=["tests.*", "tests"]),
    long_description=read('README.rst'),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "License :: Free To Use But Restricted",
        "Programming Language :: Python :: 3",
    ],
    include_package_data=True
)

# vim:sw=4:sts=4
