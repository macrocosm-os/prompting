# OpenAI Bittensor Miner
This repository contains a Bittensor Miner that uses langchain and OpenAI's model as its synapse. The miner connects to the Bittensor network, registers its wallet, and serves the GPT model to the network.

## Prerequisites

- Python 3.8+
- OpenAI Python API (https://github.com/openai/openai)

## Installation

1. Clone the repository 
```bash
git clone https://github.com/opentensor/prompting.git
```

2. Install the required packages for the [repository requirements](../../../requirements.txt) with `pip install -r requirements.txt`
3. Install the required packages for the [openai miner](requirements.txt) with `pip install -r requirements.txt`
3. Ensure that you have a `.env` file with your `OPENAI_API` key
```.env
echo OPENAI_API_KEY=YOUR-KEY > .env
```

For more configuration options related to the wallet, axon, subtensor, logging, and metagraph, please refer to the Bittensor documentation.

## Example Usage

To run the OpenAI Bittensor Miner with default settings, use the following command:

```
python3 neurons/miners/openai/miner.py
```