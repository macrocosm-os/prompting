# **VALIDATORS**

## Compute Requirements

| Resource      | Requirement       |
|---------------|-------------------|
| **VRAM**      | 62 GB             |
| **vCPU**      | 24 vCPU           |
| **RAM**       | 60 GB             |
| **Storage**   | 150 GB            |

## Installation

Clone this repository and run the [install.sh](./install.sh) script.

```bash
git clone https://github.com/opentensor/prompting.git
cd prompting
bash install.sh
```

You will also need to log into huggingface and accept the License Agreement for the LMSYS-Chat-1M dataset: https://huggingface.co/datasets/lmsys/lmsys-chat-1m :
```shell
huggingface-cli login
```

## Configuration

Before running a validator, you will need to create a .env.validator environment file. It is necessary for you to provide the following 

```text
NETUID = # [1, 61, 102]
SUBTENSOR_NETWORK = # The network name [test, main, local]
SUBTENSOR_CHAIN_ENDPOINT = # The chain endpoint [test if running on test, main if running on main, custom endpoint if running on local] 
WALLET_NAME = # Name of your wallet (coldkey) 
HOTKEY = # Name of your hotkey associated with above wallet
AXON_PORT = # Number of the open tcp port
SN19_API_KEY = # Your Subnet 19 API key
SN19_API_URL = # Your Subnet 19 API URL
OPENAI_API_KEY = # Your OpenAI API key
HF_TOKEN = # Access token from Hugging Face where you accepted the License Agreement for lmsys
```
**Example Template**: 
Use `.env.validator.example`, located in the root directory, as template for required environment variables.

## Running

After creating the above environment file, run 

```bash
bash run.sh
```
It will spawn 2 pm2 processes, one to run the validator and one to autoupdate. 

> Note: this is not an end solution, major releases or changes in requirements will still require you to manually restart the processes. Regularly monitor the health of your validator to ensure optimal performance.
