# **Miners**

# ⚠️ **DISCLAIMER** ⚠️ **DO NOT RUN THIS MINER ON MAINNET!**

> **The openai miner provided in this repo is _not intended_ to be run on mainnet!**
> 
> **If you run the base miner on mainnet, you will not earn anything!**
> It is provided as an example to help you build your own custom mining operation!
> 
## Compute Requirements

| Resource      | Requirement       |
|---------------|-------------------|
| **VRAM**      | None              |
| **vCPU**      | 24 vCPU           |
| **RAM**       | 8 GB              |
| **Storage**   | 80 GB             |

## Installation

Clone this repository and run the [install.sh](./install.sh) script.

```bash
git clone https://github.com/opentensor/prompting.git
cd prompting
bash install.sh
```

## Configuration
⚠️ **Reminder! Do not run this miner on main!** ⚠️
Before running a miner, you will need to create a .env.miner environment file. It is necessary for you to provide the following 

```text
NETUID= #[1, 61, 102]
SUBTENSOR_NETWORK= #The network name [test, main, local]
SUBTENSOR_CHAIN_ENDPOINT= #The chain endpoint [test if running on test, main if running on main, custom endpoint if running on local] 
WALLET_NAME= #Name of your wallet(coldkey) 
HOTKEY= #Name of your hotkey associated with above wallet
AXON_PORT= #Number of the open tcp port
OPENAI_API_KEY= #The openai key that you would like to mine with
```
## Testnet - RECOMMENDED

If you would like to get started with testnet, follow the directions in the discord's help-forum/Requests for Testnet Tao to test tao to register.
Then post in the Subnet 1 channel on discord so we can activate a validator for your miner to respond to.
You can use wandb to see how successful your miner would be on mainnet.

## Running

After creating the above environment file, run 

```bash
pm2 start "poetry run python neurons/miners/openai/miner.py"
```
