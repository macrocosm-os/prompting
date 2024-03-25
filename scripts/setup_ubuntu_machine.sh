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

# Install prompting package
python3 -m pip install -e .

python3 -m pip uninstall mathgenerator -y

# Install Python dependencies
python3 -m pip install -r requirements.txt

# Uninstalling uvloop to prevent conflicts with bittensor
python3 -m pip uninstall uvloop -y

# Reinstalling pydantic and transformers with specific versions that work with our repository and vllm
python3 -m pip install pydantic==1.10.7 transformers==4.36.2 angle_emb==0.3.8 peft==0.9.0

echo "Script completed successfully."
