import argparse

import bittensor as bt


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
    # TODO: Make project IPython Notebook compatible.
    # if "ipykernel" in sys.modules:
    #     # Detect if running inside IPython Notebook and filter out the Jupyter-specific arguments.
    #     args, unknown = parser.parse_known_args()
    # else:
    #     # Normal argument parsing for other environments.
    #     add_args(parser=parser)
    #     args = parser.parse_args()
    add_args(parser=parser)
    args, unknown = parser.parse_known_args()
    bt.wallet.add_args(parser)
    bt.subtensor.add_args(parser)
    bt.axon.add_args(parser)
    return bt.config(parser)
