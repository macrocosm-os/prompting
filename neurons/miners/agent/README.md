# WikiAgent Bittensor Miner
This repository contains a Bittensor Miner that uses a simple ReACT langchain agent to retrieve data from OpenAI's model alongside the wikipedia tool. The miner connects to the Bittensor network, registers its wallet, and serves the GPT model to the network.

## Prerequisites

- Python 3.8+
- OpenAI Python API (https://github.com/openai/openai)

## Installation

1. Clone the repository 
```bash
git clone https://github.com/opentensor/prompting.git
```

2. Install the required packages for the [repository requirements](../../../requirements.txt) with `pip install -r requirements.txt`
3. Install the required packages for the [wikipedia agent miner](requirements.txt) with `pip install -r requirements.txt`
3. Ensure that you have a `.env` file with your `OPENAI_API` key
```.env
echo OPENAI_API_KEY=YOUR-KEY > .env
```

For more configuration options related to the wallet, axon, subtensor, logging, and metagraph, please refer to the Bittensor documentation.

## Example Usage

To run the WikiAgent Bittensor Miner with default settings, we recommend using the model `gpt-3.5-turbo-16k` or any model with a big context window. You can run the miner using the following command:

```bash
python3 neurons/miners/wiki_agent/miner.py \
    --wallet.name <<your-wallet-name>> \
    --wallet.hotkey <<your-hotkey>>
    --neuron.model_id gpt-3.5-turbo-16k
```