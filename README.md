<picture>
    <source srcset="./assets/macrocosmos-white.png"  media="(prefers-color-scheme: dark)">
    <source srcset="./assets/macrocosmos-black.png"  media="(prefers-color-scheme: light)">
    <img src="macrocosmos-black.png">
</picture>

<div align="center">

# **Bittensor SN1** <!-- omit in toc -->
[![Discord Chat](https://img.shields.io/discord/308323056592486420.svg)](https://discord.gg/bittensor)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

---

### The Incentivized Internet <!-- omit in toc -->

[Discord](https://discord.gg/bittensor) • [Network](https://taostats.io/) • [Research](https://bittensor.com/whitepaper)

</div>

---

This repository is the **official codebase for Bittensor Subnet 1 (SN1) v1.0.0+, which was released on 22nd January 2024**. To learn more about the Bittensor project and the underlying mechanics, [read here.](https://docs.bittensor.com/).

# Introduction

This repo defines an incentive mechanism to create a distributed conversational AI for Subnet 1 (SN1).

Validators and miners are based on large language models (LLM). The validation process uses **internet-scale datasets and goal-driven behaviour to drive human-like conversations**.


</div>

# Usage

<div align="center">

**[For Validators](./assets/validator.md)** · **[For Miners](./assets/miner.md)**


</div>


# Installation

Clone this repository and run the [install.sh](./install.sh) script.

```bash
git clone https://github.com/opentensor/prompting.git
cd prompting
bash install.sh
```

If you are running a validator, you will need to install the extras via poetry.  Linux is required to run a validator.

```bash
poetry install --extras validator
poetry run pip uninstall uvloop
```

If you are running a validator, logging in to Hugging Face is required:
```shell
huggingface-cli login
```
You also need to accept the License Agreement for the LMSYS-Chat-1M dataset: https://huggingface.co/datasets/lmsys/lmsys-chat-1m

</div>

# Compute Requirements

1. To run a **validator**, you will need at least 70GB of VRAM.
2. To run the default huggingface **miner**, you will need at least 70GB of VRAM.


**It is important to note that the baseminers are not recommended for main, and exist purely as an example. Running a base miner on main will result in no emissions and a loss in your registration fee.**
If you have any questions please reach out in the SN1 channel in the Bittensor Discord.
</div>

# How to Run

To run a miner or validator, you first have to make sure you copy the .env.example file to a .env file and replace all env variables with the appropriate values. Then you can simply execute

```
poetry run python <SCRIPT_PATH>
```

where `SCRIPT_PATH` is either:
1. `neurons/miners/openai/miner.py`
2. `neurons/validator.py`

For ease of use, you can run the scripts as well with PM2, which is already installed in the install.sh file.

Example of running an Openai miner on Main:

```bash
pm2 start "poetry run python neurons/miners/openai/miner.py"
```

# Testnet
We highly recommend that you run your miners on testnet before deploying on main. This is give you an opportunity to debug your systems, and ensure that you will not lose valuable immunity time. The SN1 testnet is **netuid 61**.

In order to run on testnet, you will need to go through the same hotkey registration proceure as on main, but using **testtao**. You will need to ask for some in the community discord if you do not have any.

Then, simply set test=True in your .env file and execute all other steps as before.

## Running with autoupdate (NOTE: This is currently untested)

You can run the validator in auto-update mode by using pm2 along with the `run.sh` bash script. This command will initiate two pm2 processes: one for auto-update monitoring, named **s1_validator_update**, and another for running the validator itself, named **s1_validator_main_process**.
```bash
pm2 start run.sh --name s1_validator_autoupdate -- --wallet.name <your-wallet-name> --wallet.hotkey <your-wallet-hot-key>
```

> Note: this is not an end solution, major releases or changes in requirements will still require you to manually restart the processes. Regularly monitor the health of your validator to ensure optimal performance.
