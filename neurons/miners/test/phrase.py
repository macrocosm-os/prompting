import time
import typing
from functools import partial

from starlette.types import Send

# import base miner class which takes care of most of the boilerplate
from shared.prompting_miner import BaseStreamPromptingMiner

# Bittensor Miner Template:
from shared.protocol import StreamPromptingSynapse


class PhraseMiner(BaseStreamPromptingMiner):
    """
    This little fella responds with whatever phrase you give it.
    """

    def __init__(self, config=None):
        super().__init__(config=config)

    def forward(self, synapse: StreamPromptingSynapse) -> StreamPromptingSynapse:
        async def _forward(message: str, send: Send):
            await send(
                {
                    "type": "http.response.body",
                    "body": message,
                    "more_body": False,
                }
            )

        token_streamer = partial(_forward, "This is the phrase I'm saying!")
        return synapse.create_streaming_response(token_streamer)

    async def blacklist(self, synapse: StreamPromptingSynapse) -> typing.Tuple[bool, str]:
        return False, "All good here"

    async def priority(self, synapse: StreamPromptingSynapse) -> float:
        return 1e6


# This is the main function, which runs the miner.
if __name__ == "__main__":
    with PhraseMiner() as miner:
        while True:
            miner.log_status()
            time.sleep(5)
