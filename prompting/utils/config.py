# The MIT License (MIT)
# Copyright © 2024 Yuma Rao
# Copyright © 2023 Opentensor Foundation

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

import argparse
import bittensor as bt
import logging
from prompting import settings
from bittensor.btlogging.defines import BITTENSOR_LOGGER_NAME

logger = logging.getLogger(BITTENSOR_LOGGER_NAME)


def add_args(parser):
    """
    Adds relevant arguments to the parser for operation.
    """
    # Netuid Arg: The netuid of the subnet to connect to.
    # parser.add_argument("--netuid", type=int, help="Subnet netuid", default=1)
    parser.add_argument("--netuid", type=int, help="Subnet netuid", default=settings.NETUID)
    parser.add_argument("--wallet.name", type=str, help="Wallet name", default=settings.WALLET_NAME)
    parser.add_argument("--wallet.hotkey", type=str, help="Hotkey name", default=settings.HOTKEY)
    parser.add_argument("--subtensor.network", type=str, help="Subtensor network", default=settings.SUBTENSOR_NETWORK)


def config(cls):
    """
    Returns the configuration object specific to this miner or validator after adding relevant arguments.
    """
    parser = argparse.ArgumentParser()
    add_args(parser=parser)
    args = parser.parse_args()
    logger.info(f"RUNNING WITH ARGS: {' '.join(f'{k}={v}' for k, v in vars(args).items())}")
    bt.wallet.add_args(parser)
    bt.subtensor.add_args(parser)
    bt.logging.add_args(parser)
    bt.axon.add_args(parser)
    return bt.config(parser)
