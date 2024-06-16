#!/bin/bash

# Uninstalling mathgenerator
pip uninstall mathgenerator -y

# Installing package from the current directory
pip install -e .

# Updating the package list and installing jq and npm
sudo apt update && sudo apt install -y jq npm

# Installing PM2 globally
sudo npm install pm2 -g

# Updating PM2
pm2 update
