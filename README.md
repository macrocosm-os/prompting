
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

# SN1 Pre-Staging Testnet 102 Streaming Experiments
If you are seeing this README, you are on the pre-staging branch of SN1 designed to facilitate the development of streaming-capable miners, and running streaming experiments. The README that is currently on `main` is temporarily renamed as `README_main.md` as a reference. 

As of March 25th, 2024, SN1 will **only support miners with streaming capabilities**. Therefore, the goal of this branch is to give the community access to the production environment before it goes live. 

**The intended use of this branch is for miners and validators to run on testnet 102**. 

## Important Questions

### 1. What is streaming? 
Streaming is when the miner sends back chunks of data to the validator to enable getting partial responses before timeout. The benefits are two fold:
1. getting rewards for unfinished, high quality responses, and
2. enabling a stream-based UI for responsive frontends. 

### 2. How will this change my miner? 
Stream miners need to implement a new `forward` method that enables async communications to the validator. The template for miners can be found in `docs/stream_miner_template.md`. Follow the instructions in this markdown file to learn the **important** steps needed to convert your miner to this new format. Alternatively, you can run one of the two base miners in the repo: 

```bash
# To run the miner
python <MINER_PATH>
    --netuid 102
    --subtensor.network test
    --wallet.name <your miner wallet> # Must be created using the bittensor-cli
    --wallet.hotkey <your validator hotkey> # Must be created using the bittensor-cli
    --logging.debug # Run in debug mode, alternatively --logging.trace for trace mode
    --axon.port #VERY IMPORTANT: set the port to be one of the open TCP ports on your machine
```
where `MINER_PATH` is either: 
1. neurons/miners/huggingface/miner.py
2. neurons/miners/openai/miner.py

It is highly important that you set the `--axon.port` to be one of the open TCP ports on your machine. If you do not do this, you will not recieve requests from validators and will not be able to check the performance of your miner. 

### 3. Do I need to change my hardware?!? 
No! The hardware requirements for running streaming are identical to before, so there is no need to provision new or more capable resources. 

### 4. Registration 
Similar to registering a miner on mainnet, you need to register your miner for testnet 102. Here is a simple command to do so: 
`btcli subnet register --netuid 102 --subtensor.network test --wallet.name <YOUR_WALLET_NAME> --wallet.hotkey <YOUR_HOTKEY_NAME>`

To register, you will need test tao! Please reach out to @mccrinbc or @ellesmier for test tao if needed. 

### 5. Validators
Folks who want to run validators are encouraged to do so. The SN1 development team are dedicating resources to run (at present) 2 validators on testnet 102 to ensure that miners will be busy getting queried. 

### 6. How do I know that my miner is working? 
The easiest way to make sure that your miner is working is to use the script in `scripts/client.py`. You can query your miner from a **separate** registered hotkey. This must be done because the same hotkey cannot ping itself. 

```bash
python scripts/client.py --uids YOUR_UIDS --wallet_name <YOUR_WALLET_NAME> --hotkey <YOUR_DIFFERENT_HOTKEY> --message "WHATEVER MESSAGE YOU WANT TO SEND"
```

An example is:
```bash
python scripts/client.py --wallet_name testnet --hotkey my_validator --message "Write me a 500 word essay about albert einstein" --uids 1 2
```

In order to query the networek, the hotkey you provide will need to have a vpermit. Until netuid 102 has more than 64 neurons registered, all neurons will have a vpermit by default. To check your neurons on netuid 102, run the following btcli command: 

```bash
btcli wallet overview --subtensor.network test --wallet.name <YOUR_WALLET_NAME> 
```

You can also wait to get queried by the validators, and pull the appropriate wandb data to checkout that your miners are being queried will all the data needed for analysis. Filtering by `netuid=102` is important here so that you see the active validators running on testnet, and previous test data. 

[Click here to be taken to the wandb validator page filtered by netuid 102.](https://wandb.ai/opentensor-dev/alpha-validators?nw=nwusermccrinbcsl) 

### 7. Installation
You will need to reinstall the repo to ensure that you do not see any errors. Please run:
```bash
python -m pip install -e . --force-reinstall
```

> Important: vLLM currently faces a [notable limitation](https://github.com/vllm-project/vllm/issues/3012) in designating a specific GPU for model execution via code. Consequently, to employ a particular CUDA device for your model's operations, it's necessary to manually adjust your environment variable `CUDA_VISIBLE_DEVICES`. For instance, setting `export CUDA_VISIBLE_DEVICES=1,2` will explicitly define the CUDA devices available for use.

# Real-time monitoring with wandb integration

Check out real-time public logging by looking at the project [here](https://wandb.ai/opentensor-dev/alpha-validators)

## License
This repository is licensed under the MIT License.
```text
# The MIT License (MIT)
# Copyright © 2024 Yuma Rao

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
