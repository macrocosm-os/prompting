from prompting import settings
settings.settings = settings.Settings(mode="validator")
settings = settings.settings

import argparse
import asyncio

import bittensor as bt
import time
from loguru import logger

from typing import List, Awaitable
from prompting.base.protocol import StreamPromptingSynapse
from prompting.settings import settings

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

assert settings.TEST_MINER_IDS, "Please provide the miner ids to query in the .env.validator file as variable TEST_MINER_IDS"


async def handle_response(responses: list[Awaitable]) -> List[str]:
    synapses = []

    for uid, response in zip(settings.TEST_MINER_IDS, responses):
        chunk_times = []
        start_time = time.time()

        i = 0
        async for chunk in response:
            chunk_times.append(-start_time + (start_time := time.time()))
            logger.info(f"UID: {uid}. chunk {(i := i + 1)} ({chunk_times[-1]:.3f}s) for resp: {chunk}")

        logger.success(f"UID {uid} took {sum(chunk_times):.3f} seconds\n")
        if not isinstance(chunk, bt.Synapse):
            raise Exception(f"Last object yielded is not a synapse; the miners response did not finish: {chunk}")
        synapses.append(chunk)  # last object yielded is the synapse itself with completion filled

    return synapses


async def query_stream_miner(
    synapse_protocol: bt.Synapse,
    message: str | None = None,
):
    if message is None:
        message = "Give me some information about the night sky."

    synapse = synapse_protocol(
        roles=["user"],
        messages=[message],
        task_name = 'inference'
    )
    dendrite = bt.dendrite(wallet=settings.WALLET)

    logger.info(f"Synapse: {synapse}")

    try:
        axons = [settings.METAGRAPH.axons[uid] for uid in settings.TEST_MINER_IDS]
        responses = await dendrite(  # responses is an async generator that yields the response tokens
            axons,
            synapse,
            deserialize=False,
            timeout=settings.NEURON_TIMEOUT,
            streaming=True,
        )

        return await handle_response(responses)

    except Exception as e:
        logger.exception(e)
        logger.error(f"Exception during query to uids: {settings.TEST_MINER_IDS}: {e}")
        return None


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Query a Bittensor synapse with given parameters.")

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
            synapse_protocol=StreamPromptingSynapse,
            message=args.message,
        )
    )
