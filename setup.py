#!/usr/bin/env python3

"""Python setuptools setup.

"""

import os
from setuptools import setup


# Get current dir
current_dir = os.path.dirname(os.path.realpath(__file__))

# Get path to requirements file
requirements_path = os.path.abspath(os.path.join(
    current_dir, "requirements.txt"))

# Get required packages
required_packages = [
    line.strip()
    for line in open(requirements_path, "r")
    if line.strip() and not line.lstrip().startswith("#")
]

# Set up SMEAGOL
setup(
    name="smeagol-bio",
    version="0.1.0",
    install_requires=required_packages,
    include_package_data=True,
    packages=["smeagol"],
    python_requires=">=3.7, !=3.9",
    platforms=["any"],
)
