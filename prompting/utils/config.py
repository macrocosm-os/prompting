import argparse
import bittensor as bt
from loguru import logger


def add_args(parser):
    """
    Adds relevant arguments to the parser for operation.
    """
    # Netuid Arg: The netuid of the subnet to connect to.
    # parser.add_argument("--netuid", type=int, help="Subnet netuid", default=1)
    parser.add_argument("--netuid", type=int, help="Subnet netuid")
    parser.add_argument("--wallet.name", type=str, help="Wallet name")
    parser.add_argument("--wallet.hotkey", type=str, help="Hotkey name")
    parser.add_argument("--subtensor.network", type=str, help="Subtensor network")
    parser.add_argument("--axon.port", type=int, help="The open port to run on")


def config() -> bt.config:
    """
    Returns the configuration object specific to this miner or validator after adding relevant arguments.
    """
    parser = argparse.ArgumentParser()
    add_args(parser=parser)
    args = parser.parse_args()
    logger.info(f"RUNNING WITH ARGS: {' '.join(f'{k}={v}' for k, v in vars(args).items())}")
    bt.wallet.add_args(parser)
    bt.subtensor.add_args(parser)
    bt.axon.add_args(parser)
    return bt.config(parser)
