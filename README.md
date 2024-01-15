
<div align="center">

# **Bittensor Text-PromptingSynapse** <!-- omit in toc -->
[![Discord Chat](https://img.shields.io/discord/308323056592486420.svg)](https://discord.gg/bittensor)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT) 

---

### The Incentivized Internet <!-- omit in toc -->

[Discord](https://discord.gg/bittensor) • [Network](https://taostats.io/) • [Research](https://bittensor.com/whitepaper)

</div>

---

This template contains all the necessary files and functions to run Bittensor's Text-PromptingSynapse Subnet. You can try running miners on netuid 8 in Bittensor's test network.

# Introduction

[To learn more about the Bittensor project and the underlying mechanics, read here.](https://docs.bittensor.com/)

This repository is a subnet for text prompting with large language models (LLM). Inside, you will find miners and validators designed by the Subnet 1 Development team to validate and serve language models. The current validator implementation uses internet-scale datasets and goal-driven agent behavior to drive human-like conversations. 

</div>

---

# Installation
This repository requires python3.8 or higher. To install, simply clone this repository and install the requirements.
```bash
git clone https://github.com/opentensor/prompting.git
cd prompting
python -m pip install -r requirements.txt
python -m pip install -e .
```

</div>

---

Currently, the incentive mechanism and miners are a work in progress, and should only be run on the test chain. If you require test tao, please reach out to ____.

Prior to running a miner or validator, you must [create a wallet](https://github.com/opentensor/docs/blob/main/reference/btcli.md) and [register the wallet to a netuid](https://github.com/opentensor/docs/blob/main/subnetworks/registration.md). Once you have done so, you can run the miner and validator with the following commands.
```bash
# To run the miner
python -m neurons/miners/bittensorLM/miner.py 
    --netuid 61
    --subtensor.network test 
    --wallet.name <your miner wallet> # Must be created using the bittensor-cli
    --wallet.hotkey <your validator hotkey> # Must be created using the bittensor-cli
    --logging.debug # Run in debug mode, alternatively --logging.trace for trace mode

# To run the validator
python neurons/validator.py
    --netuid 61
    --subtensor.network test 
    --wallet.name <your validator wallet>  # Must be created using the bittensor-cli
    --wallet.hotkey <your validator hotkey> # Must be created using the bittensor-cli
    --logging.debug # Run in debug mode, alternatively --logging.trace for trace mode
```

</div>

---


# Running

Running instructions for main chain are coming soon!


# Real-time monitoring with wandb integration

A fresh wandb project is on its way!

## License
This repository is licensed under the MIT License.
```text
# The MIT License (MIT)
# Copyright © 2023 Yuma Rao

# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
# documentation files (the “Software”), to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software,
# and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all copies or substantial portions of
# the Software.

# THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO
# THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
# OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.
```
