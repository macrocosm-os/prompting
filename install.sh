#!/bin/bash

# Install poetry
pip install poetry

# Set the destination of the virtual environment to the project directory
poetry config virtualenvs.in-project true

# Install the project dependencies
poetry install

# Build AutoAWQ==0.2.7.post3 from source
git clone https://github.com/casper-hansen/AutoAWQ.git
cd AutoAWQ && git checkout cbd6a75b065e94a3e530dfdbb8f3973f0d954ec0 && poetry run pip install -e . && cd ..
rm -r AutoAWQ

poetry run pip install flash-attn --no-build-isolation

# Check if jq is installed and install it if not
if ! command -v jq &> /dev/null
then
    apt update && apt install -y jq
fi

# Check if npm is installed and install it if not
if ! command -v npm &> /dev/null
then
    apt update && apt install -y npm
fi

# Check if pm2 is installed and install it if not
if ! command -v pm2 &> /dev/null
then
    npm install pm2 -g
fi
