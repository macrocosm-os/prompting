# OpenAI Bittensor Miner

This repository contains a Bittensor Miner that uses LangChain and OpenAI's model as its synapse. The miner connects to the Bittensor network, registers its wallet, and serves the GPT model to the network.

## Prerequisites

- OpenAI API Key (if you would like to run the OpenAI demo miner)
- Python and pip installed

## Installation

1. **Clone the repository**:

    ```bash
    git clone https://github.com/opentensor/prompting.git
    ```

2. **Install all Python packages**:

    ```bash
    bash install.sh
    ```

3. You then need to activate the virtual environment (managed by poetry) by running
    ```bash
    poetry shell
    ```

4. **Set up your .env file with your OpenAI API key**:

    ```bash
    echo OPENAI_API_KEY=YOUR-KEY > .env
    ```

5. **Set up your wallet(s)**:
    - The `run_miner.sh` and `run_validator.sh` scripts assume your wallets are called `miner` and `validator` with the hotkeys `miner_hotkey` and `validator_hotkey` respectively. You may modify these files to match your wallet names.
    - Once your wallets are set up and registered to the testnet (see [Bittensor documentation](https://docs.bittensor.com/) for how to do this), you can execute the validator/miner using the `run_miner.sh` and `run_validator.sh` scripts.

6. **Query the miner**:
    - If you have a miner running, you can use the `client.py` file to query your miner and get responses:

    ```bash
    python client.py
    ```

For more configuration options related to the wallet, axon, subtensor, logging, and metagraph, please refer to the [Bittensor documentation](https://docs.bittensor.com/).

---

Feel free to reach out if you have any questions or need further assistance. You can reach us through the [bittensor discord](https://discord.gg/UqAxyhrf) (subnet 1 channel) or via email (felix.quinque(at)macrocosmos.ai)
