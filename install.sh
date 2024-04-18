#!/bin/bash

# Uninstalling mathgenerator
pip uninstall mathgenerator -y

# Installing package from the current directory
pip install -e .
