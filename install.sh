#!/bin/bash

# Uninstalling mathgenerator
pip uninstall mathgenerator -y

# Installing package from the current directory
pip install -e .

# Install AutoAWQ without dependencies, to avoid conflicts with lower version of transformers.
pip install zstandard --no-deps
pip installautoawq-kernels --no-deps
pip install autoawq==0.2.5 --no-deps
