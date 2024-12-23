#!/bin/bash

# Install poetry if not already installed
if ! command -v poetry &> /dev/null
then
    echo "Poetry is not installed. Installing Poetry..."
    pip install poetry
fi

# Install project dependencies
poetry install

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

# Define the PM2 configuration
echo "module.exports = {
  apps: [
    {
      name: 'api_server',
      script: 'poetry',
      interpreter: 'none',
      args: ['run', 'python', 'validator_api/api.py'],
      min_uptime: '5m',
      max_restarts: 5
    }
  ]
};" > api.config.js

# Start the API server using PM2
pm2 start api.config.js

pm2 log api_server
