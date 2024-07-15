#!/bin/bash

# Uninstalling mathgenerator
pip uninstall mathgenerator -y

# Installing package from the current directory
pip install -e .

# Updating the package list and installing jq and npm
apt update && apt install -y jq npm

# Installing PM2 globally
npm install pm2 -g

# Updating PM2
pm2 update

# installing uvloop as it causes issues with threading (it's part of the requirements for bt but never used)
pip uninstall uvloop -y