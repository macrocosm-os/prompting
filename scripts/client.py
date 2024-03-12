import argparse
import asyncio

import bittensor as bt
import time

from typing import List, Awaitable
from prompting.protocol import StreamPromptingSynapse

"""
This has assumed you have:
1. Registered your miner on the chain (finney/test)
2. Are serving your miner on an open port (e.g. 12345)

Steps:
- Instantiate your synapse subclass with the relevant information. E.g. messages, roles, etc.
- Instantiate your wallet and a dendrite client
- Query the dendrite client with your synapse object
- Iterate over the async generator to extract the yielded tokens on the server side
"""


async def handle_response(
    uids: List[int], responses: List[Awaitable]
) -> tuple[str, str]:
    synapses = []

    for uid_num, resp in enumerate(responses):
        ii = 0
        chunk_times = []
        start_time = time.time()

        chunk_start_time = time.time()
        async for chunk in resp:
            chunk_time = round(time.time() - chunk_start_time, 3)
            bt.logging.info(
                f"UID: {uids[uid_num]}. chunk {ii}({chunk_time}s) for resp: {chunk} "
            )
            ii += 1

            chunk_times.append(chunk_time)
            chunk_start_time = time.time()

        bt.logging.success(
            f"UID {uids[uid_num]} took {(time.time() - start_time):.3f} seconds\n"
        )

        synapse = (
            chunk  # last object yielded is the synapse itself with completion filled
        )
        synapses.append(synapse)

    return synapses


async def query_stream_miner(
    args, synapse_protocol, wallet_name, hotkey, network, netuid, message=None
):
    if message is None:
        message = "Give me some information about the night sky."

    syn = synapse_protocol(
        roles=["user"],
        messages=[message],
    )

    # create a wallet instance with provided wallet name and hotkey
    wallet = bt.wallet(name=wallet_name, hotkey=hotkey)

    # instantiate the metagraph with provided network and netuid
    metagraph = bt.metagraph(netuid=netuid, network=network, sync=True, lite=False)

    # Create a Dendrite instance to handle client-side communication.
    dendrite = bt.dendrite(wallet=wallet)

    bt.logging.info(f"Synapse: {syn}")

    async def main():
        try:
            uids = args.uids
            axons = [metagraph.axons[uid] for uid in uids]
            responses = await dendrite(  # responses is an async generator that yields the response tokens
                axons,
                syn,
                deserialize=False,
                timeout=10,
                streaming=True,
            )

            return await handle_response(uids, responses)

        except Exception as e:
            bt.logging.error(f"Exception during query to uids: {uids}: {e}")
            return None

    await main()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Query a Bittensor synapse with given parameters."
    )

    parser.add_argument(
        "--uids",
        nargs="+",
        required=True,
        help="UIDs to query.",
        default=[1, 2],
        type=int,
    )
    parser.add_argument("--netuid", type=int, default=102, help="Network Unique ID")
    parser.add_argument(
        "--wallet_name", type=str, default="default", help="Name of the wallet"
    )
    parser.add_argument(
        "--hotkey", type=str, default="default", help="Hotkey for the wallet"
    )
    parser.add_argument(
        "--network",
        type=str,
        default="test",
        help='Network type, e.g., "test" or "mainnet"',
    )

    parser.add_argument(
        "--message",
        type=str,
        default="What is the meaning of life?",
        help="A question that you want to ask to the ai",
    )

    # Parse arguments
    args = parser.parse_args()

    # Running the async function with provided arguments
    asyncio.run(
        query_stream_miner(
            args,
            synapse_protocol=StreamPromptingSynapse,
            wallet_name=args.wallet_name,
            hotkey=args.hotkey,
            network=args.network,
            netuid=args.netuid,
            message=args.message,
        )
    )
