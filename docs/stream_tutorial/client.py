import argparse
import asyncio
import bittensor as bt

from typing import List, Awaitable
import pdb

from prompting.protocol import StreamPromptingSynapse, PromptingSynapse

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


def setup(synapse_protocol, wallet_name, hotkey, network, netuid, uid):
    syn = synapse_protocol(
        roles=["user"],
        messages=[
            "hello this is a test of a streaming response. Lorem ipsum dolor sit amet, consectetur adipiscing elit, sed do eiusmod tempor incididunt ut labore et dolore magna aliqua."
        ],
    )

    # create a wallet instance with provided wallet name and hotkey
    wallet = bt.wallet(name=wallet_name, hotkey=hotkey)

    # instantiate the metagraph with provided network and netuid
    metagraph = bt.metagraph(netuid=netuid, network=network, sync=True, lite=False)

    # Grab the axon you're serving
    axon = metagraph.axons[uid]

    # Create a Dendrite instance to handle client-side communication.
    dendrite = bt.dendrite(wallet=wallet)

    print("metagraph.axons: ", metagraph.axons)
    print("axon: ", axon)
    print("dendrite: ", dendrite)

    print(f"Synapse: {syn}")
    print(f"Synapse type: {syn.__class__.__name__}")

    return syn, dendrite, metagraph


async def handle_response(uid: str, responses: List[Awaitable]) -> tuple[str, str]:
    full_response = ""
    ii = 0
    for resp in responses:
        # pdb.set_trace(header="inside handle_response")
        async for chunk in resp:
            print(f"\nchunk for resp {ii}: {chunk}", end="", flush=True)
            # pdb.set_trace(header="\nCheck chunk")

        ii += 1 

        synapse = (
            chunk  # last object yielded is the synapse itself with completion filled
        )

        print(f"Final Synapse: {synapse}")
        break
    return uid, full_response


async def query_stream_miner(
    synapse_protocol, wallet_name, hotkey, network, netuid, uid
):
    syn = synapse_protocol(
        roles=["user"],
        messages=["Give me some information about the night sky."],
    )

    # create a wallet instance with provided wallet name and hotkey
    wallet = bt.wallet(name=wallet_name, hotkey=hotkey)

    # instantiate the metagraph with provided network and netuid
    metagraph = bt.metagraph(netuid=netuid, network=network, sync=True, lite=False)

    # Grab the axon you're serving
    axon = metagraph.axons[uid]

    # Create a Dendrite instance to handle client-side communication.
    dendrite = bt.dendrite(wallet=wallet)

    print("dendrite: ", dendrite)

    print(f"Synapse: {syn}")
    print(f"Synapse type: {syn.__class__.__name__}")

    async def main():
        try:
            responses = await dendrite(  # responses is an async generator that yields the response tokens
                [metagraph.axons[uid]],
                syn,
                deserialize=False,
                timeout=20,
                streaming=True,
            )
            # return responses
            return await handle_response(uid, responses)

        except Exception as e:
            print(f"Exception during query for uid {uid}: {e}")
            return uid, None

    await main()


def query_miner(synapse_protocol, wallet_name, hotkey, network, netuid, uid):
    syn = synapse_protocol(
        roles=["user"],
        messages=[
            "hello this is a test of a streaming response. Lorem ipsum dolor sit amet, consectetur adipiscing elit, sed do eiusmod tempor incididunt ut labore et dolore magna aliqua."
        ],
    )

    # create a wallet instance with provided wallet name and hotkey
    wallet = bt.wallet(name=wallet_name, hotkey=hotkey)

    # instantiate the metagraph with provided network and netuid
    metagraph = bt.metagraph(netuid=netuid, network=network, sync=True, lite=False)

    # Grab the axon you're serving
    axon = metagraph.axons[uid]

    # Create a Dendrite instance to handle client-side communication.
    dendrite = bt.dendrite(wallet=wallet)

    print("metagraph.axons: ", metagraph.axons)
    print("axon: ", axon)
    print("dendrite: ", dendrite)

    print(f"Synapse: {syn}")
    print(f"Synapse type: {syn.__class__.__name__}")

    try:
        responses = dendrite(
            [metagraph.axons[uid]],
            syn,
            deserialize=False,
            timeout=20,
            streaming=False,
        )
        return responses

    except Exception as e:
        print(f"Exception during query for uid {uid}: {e}")
        return uid, None


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Query a Bittensor synapse with given parameters."
    )

    # Adding arguments
    parser.add_argument(
        "--uid",
        type=int,
        required=True,
        help="Your unique miner ID on the chain",
    )
    parser.add_argument("--netuid", type=int, required=True, help="Network Unique ID")
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

    # Parse arguments
    args = parser.parse_args()

    # Running the async function with provided arguments
    asyncio.run(
        query_stream_miner(
            synapse_protocol=StreamPromptingSynapse,
            wallet_name=args.wallet_name,
            hotkey=args.hotkey,
            network=args.network,
            netuid=args.netuid,
            uid=args.uid,
        )
    )
