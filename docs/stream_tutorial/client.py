import argparse
import asyncio
import bittensor as bt

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


async def handle_response(uid: str, responses) -> tuple[str, str]:
    full_response = ""
    for resp in responses:
        async for chunk in resp:
            if isinstance(chunk, str):
                bt.logging.trace(chunk)
                full_response += chunk
        bt.logging.debug(f"full_response for uid {uid}: {full_response}")
        break
    return uid, full_response


async def query_stream_miner(syn, dendrite, metagraph, uid):
    if syn.__class__.__name__ == "StreamPromptingSynapse":
        streaming = True
    else:
        streaming = False

    try:
        responses = await dendrite(
            [metagraph.axons[uid]],
            syn,
            deserialize=False,
            timeout=20,
            streaming=streaming,
        )
        print(f"Responses: {responses}")
        return responses
    #     return await handle_response(uid, responses)

    except Exception as e:
        print(f"Exception during query for uid {uid}: {e}")
        return uid, None


def query_miner(syn, dendrite, metagraph, uid):
    if syn.__class__.__name__ == "StreamPromptingSynapse":
        streaming = True
    else:
        streaming = False

    try:
        responses = dendrite(
            [metagraph.axons[uid]],
            syn,
            deserialize=False,
            timeout=20,
            streaming=streaming,
        )
        return responses
    #     return await handle_response(uid, responses)

    except Exception as e:
        print(f"Exception during query for uid {uid}: {e}")
        return uid, None


async def query_synapse(my_uid, wallet_name, hotkey, network, netuid):
    syn = StreamPrompting(
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
    axon = metagraph.axons[my_uid]

    # Create a Dendrite instance to handle client-side communication.
    dendrite = bt.dendrite(wallet=wallet)

    async def main():
        responses = await dendrite([axon], syn, deserialize=False, streaming=True)

        for resp in responses:
            i = 0
            async for chunk in resp:
                i += 1
                if i % 5 == 0:
                    print()
                if isinstance(chunk, list):
                    print(chunk[0], end="", flush=True)
                else:
                    # last object yielded is the synapse itself with completion filled
                    synapse = chunk
            break

    # Run the main function with asyncio
    await main()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Query a Bittensor synapse with given parameters."
    )

    # Adding arguments
    parser.add_argument(
        "--my_uid",
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
        query_synapse(
            args.my_uid,
            args.wallet_name,
            args.hotkey,
            args.network,
            args.netuid,
        )
    )
