import pytest
import bittensor as bt
import asyncio

from typing import List, AsyncGenerator
from prompting.mock import MockDendrite, MockMetagraph, MockSubtensor
from prompting.protocol import StreamPromptingSynapse


def compare_with_threshold(timeout: float, process_time: float, threshold: float):
    """
    Compare two numbers where b cannot be greater than a by the given threshold.

    Parameters:
        threshold (float): between 0 and 1 to represent percentage of timout.

    Returns:
        bool: True if b is less than or equal to a + threshold, False otherwise.
    """
    return process_time <= timeout * (1 + threshold)


async def handle_response(responses) -> List[StreamPromptingSynapse]:
    synapses = []
    resp_idx = 0
    for resp in responses:
        ii = 0
        resp_idx += 1
        async for chunk in resp:
            print(f"\nchunk {ii} for resp {resp_idx}: {chunk}", end="", flush=True)
            ii += 1

        synapse = (
            chunk  # last object yielded is the synapse itself with completion filled
        )

        synapses.append(synapse)
    return synapses


@pytest.mark.parametrize("timeout", [0.1])
def test_mock_streaming(timeout: float):
    netuid = 1

    mock_wallet = bt.MockWallet()
    mock_dendrite = MockDendrite(wallet=mock_wallet)
    mock_subtensor = MockSubtensor(netuid=netuid, wallet=mock_wallet)
    mock_metagraph = MockMetagraph(netuid=netuid, subtensor=mock_subtensor)

    streaming = True
    messages = [
        "Einstein's famous equation, E=mc^2, showed the equivalence of mass and energy and laid the foundation for the development of nuclear energy. His work also contributed to the understanding of quantum mechanics and the photoelectric effect, for which he was awarded the Nobel Prize in Physics in 1921."
    ]

    synapse = StreamPromptingSynapse(
        roles=["user"],
        messages=messages,
        max_tokens=256,
    )

    async def get_responses(
        synapse: StreamPromptingSynapse, timeout: float
    ) -> List[AsyncGenerator]:
        return await mock_dendrite(  # responses is an async generator that yields the response tokens
            axons=mock_metagraph.axons,
            synapse=synapse,
            deserialize=False,
            timeout=timeout,
            streaming=streaming,
        )

    responses = asyncio.run(get_responses(synapse=synapse, timeout=timeout))

    async def main(responses: List[AsyncGenerator]):
        try:
            return await handle_response(responses)

        except Exception as e:
            print(f"Error: {e}")

    synapses = asyncio.run(main(responses=responses))
    ansr = synapse.messages[-1]

    threshold = 0.2

    for syn in synapses:
        if syn.dendrite.status_code == 200:
            assert syn.completion == ansr
            assert compare_with_threshold(
                process_time=syn.dendrite.process_time,
                timeout=timeout,
                threshold=threshold,
            ), f"Process time ({syn.dendrite.process_time}) is greater than timeout w/ threshold ({timeout * (1 + threshold)})."
        elif syn.dendrite.status_code == 408:
            assert len(syn.completion) < len(ansr)
            assert syn.dendrite.process_time == timeout
