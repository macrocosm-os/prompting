<picture>
    <source srcset="./assets/macrocosmos-white.png"  media="(prefers-color-scheme: dark)">
    <img src="macrocosmos-white.png">
</picture>

<picture>
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

# Installation
This repository requires python3.9 or higher. To install it, simply clone this repository and run the [install.sh](./install.sh) script.
```bash
git clone https://github.com/opentensor/prompting.git
cd prompting
bash install.sh
```

</div>

# Compute Requirements

1. To run a **validator**, you will need at least 24GB of VRAM. 
2. To run the default huggingface **miner**, you will need at least 18GB of VRAM. 

</div>

# How to Run
You can use the following command to run a miner or a validator. 

```bash
python <SCRIPT_PATH>
    --netuid 1
    --subtensor.network <finney/local/test>
    --neuron.device cuda
    --wallet.name <your wallet> # Must be created using the bittensor-cli
    --wallet.hotkey <your hotkey> # Must be created using the bittensor-cli
    --logging.debug # Run in debug mode, alternatively --logging.trace for trace mode
    --axon.port # VERY IMPORTANT: set the port to be one of the open TCP ports on your machine
```

where `SCRIPT_PATH` is either: 
1. neurons/miners/huggingface/miner.py
2. neurons/miners/openai/miner.py
3. neurons/validator.py

For ease of use, you can run the scripts as well with PM2. Installation of PM2 is: 
**On Linux**:
```bash
sudo apt update && sudo apt install jq && sudo apt install npm && sudo npm install pm2 -g && pm2 update
``` 

Example of running a Llama3 miner: 
```bash
pm2 start neurons/miners/huggingface/miner.py --interpreter python3 --name solar_miner -- --netuid 1  --subtensor.network finney --wallet.name my_wallet --wallet.hotkey m1 --neuron.model_id MaziyarPanahi/Meta-Llama-3-70B-Instruct-GPTQ --axon.port 21988 --logging.debug
``` 

# Testnet 
We highly recommend that you run your miners on testnet before deploying on main. This is give you an opportunity to debug your systems, and ensure that you will not lose valuable immunity time. The SN1 testnet is **netuid 61**. 

In order to run on testnet, you will need to go through the same hotkey registration proceure as on main, but using **testtao**. You will need to ask for some in the community discord if you do not have any. 

To run:

```bash
pm2 start neurons/miners/huggingface/miner.py --interpreter python3 --name solar_miner -- --netuid 61  --subtensor.network test --wallet.name my_test_wallet --wallet.hotkey m1 --neuron.model_id MaziyarPanahi/Meta-Llama-3-70B-Instruct-GPTQ --axon.port 21988 --logging.debug 
```

# Limitations
> Important: vLLM currently faces a [notable limitation](https://github.com/vllm-project/vllm/issues/3012) in designating a specific GPU for model execution via code. Consequently, to employ a particular CUDA device for your model's operations, it's necessary to manually adjust your environment variable `CUDA_VISIBLE_DEVICES`. For instance, setting `export CUDA_VISIBLE_DEVICES=1,2` will explicitly define the CUDA devices available for use.

# Resources
The archiecture and methodology of SN1 is complex, and as such we have created a comprehensive resource to outline our design. Furthermore, we have strict requirements for how miners should interact with the network. Below are the currently available resources for additional information: 

1. [SN1 architecture details](docs/SN1_validation.md)
2. [StreamMiner requirements](docs/stream_miner_template.md)
