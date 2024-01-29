# Zephyr Bittensor Miner
This repository contains a Bittensor Miner that uses [HuggingFaceH4/zephyr-7b-beta](https://huggingface.co/HuggingFaceH4/zephyr-7b-beta). The miner connects to the Bittensor network, registers its wallet, and serves the zephyr model to the network.

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
```

For more configuration options related to the wallet, axon, subtensor, logging, and metagraph, please refer to the Bittensor documentation.

## Example Usage

To run the Zephyr Bittensor Miner with default settings, use the following command:
```bash
python3 neurons/miners/zephyr/miner.py \
    --wallet.name <<your-wallet-name>> \
    --wallet.hotkey <<your-hotkey>>
    --neuron.model_id HuggingFaceH4/zephyr-7b-beta
```

You will need 18GB of GPU to run this miner in comfortable settings.

You can also run the quantized version of this model that takes ~10GB of GPU RAM by adding the flag `--neuron.load_quantized`:
```bash
python3 neurons/miners/zephyr/miner.py \
    --wallet.name <<your-wallet-name>> \
    --wallet.hotkey <<your-hotkey>>
    --neuron.model_id HuggingFaceH4/zephyr-7b-beta
    --neuron.load_quantized True
```