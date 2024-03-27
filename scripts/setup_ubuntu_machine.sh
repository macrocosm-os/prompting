#!/bin/bash


apt-get update
apt-get install sudo

# Install python3-pip
sudo apt install -y python3-pip

# Upgrade bittensor
python3 -m pip install --upgrade bittensor

apt install tree

# Install required packages
sudo apt update && sudo apt install jq && sudo apt install npm && sudo npm install pm2 -g && pm2 update

# echo 'export OPENAI_API_KEY=YOUR_OPEN_AI_KEY' >> ~/.bashrc

# Clone the repository
git clone https://github.com/opentensor/prompting.git

# Change to the prompting directory
cd prompting

bash install.sh

echo "Script completed successfully."
