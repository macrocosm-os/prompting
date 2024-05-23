#!/bin/bash

# Uninstalling mathgenerator
pip uninstall mathgenerator -y

# Installing package from the current directory
pip install -e .

# Miner requirements: AutoAWQ without dependencies, to avoid conflicts with other modules.
pip install zstandard --no-deps
pip install autoawq-kernels --no-deps
pip install autoawq==0.2.5 --no-deps
