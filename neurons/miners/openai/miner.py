import time
import bittensor as bt
import os
from starlette.types import Send
from functools import partial
from typing import Awaitable

# Bittensor Miner Template:
from prompting.base.prompting_miner import BaseStreamPromptingMiner
from prompting.base.protocol import StreamPromptingSynapse

# import base miner class which takes care of most of the boilerplate

from neurons.miners.openai.utils import OpenAIUtils
from openai import OpenAI
from traceback import print_exception

from prompting import settings


class OpenAIMiner(BaseStreamPromptingMiner, OpenAIUtils):
    """Langchain-based miner which uses OpenAI's API as the LLM.
    This miner does not use any tools or external APIs when processing requests - it relies entirely on the models' own representation and world model. In some cases, this can produce lower quality results.
        You should also install the dependencies for this miner, which can be found in the requirements.txt file in this directory.
    """

    def __init__(self, config=None):
        super().__init__(config=config)

        bt.logging.info(f"Initializing with model {settings.NEURON_MODEL_ID_MINER}...")

        if settings.WANDB_ON:
            self.identity_tags = ("openai_miner",) + (settings.NEURON_MODEL_ID_MINER)
        api_key = os.environ.get("OPENAI_API_KEY")

        # Set openai key and other args
        self.model = OpenAI(api_key=api_key)

        self.system_prompt = settings.NEURON_SYSTEM_PROMPT
        self.accumulated_total_tokens = 0
        self.accumulated_prompt_tokens = 0
        self.accumulated_completion_tokens = 0
        self.accumulated_total_cost = 0

    def forward(self, synapse: StreamPromptingSynapse) -> Awaitable:
        async def _forward(
            self,
            synapse: StreamPromptingSynapse,
            init_time: float,
            timeout_threshold: float,
            send: Send,
        ):
            buffer = []
            accumulated_chunks = []
            accumulated_chunks_timings = []
            messages = []
            temp_completion = ""  # for wandb logging
            timeout_reached = False

            try:
                system_prompt_message = [{"role": "system", "content": self.system_prompt}]
                synapse_messages = [
                    {"role": role, "content": message} for role, message in zip(synapse.roles, synapse.messages)
                ]

                messages = system_prompt_message + synapse_messages

                start_time = time.time()
                stream_response = self.model.chat.completions.create(
                    model=settings.NEURON_MODEL_ID_MINER,
                    messages=messages,
                    temperature=settings.NEURON_TEMPERATURE,
                    max_tokens=settings.NEURON_MAX_TOKENS,
                    stream=True,
                )

                for chunk in stream_response:
                    chunk_content = chunk.choices[0].delta.content

                    if chunk_content is None:
                        bt.logging.info("OpenAI returned chunk content with None")
                        continue

                    accumulated_chunks.append(chunk_content)
                    accumulated_chunks_timings.append(time.time() - start_time)

                    buffer.append(chunk_content)

                    if time.time() - init_time > timeout_threshold:
                        bt.logging.debug("⏰ Timeout reached, stopping streaming")
                        timeout_reached = True
                        break

                    if len(buffer) == settings.NEURON_STREAMING_BATCH_SIZE:
                        joined_buffer = "".join(buffer)
                        temp_completion += joined_buffer
                        bt.logging.debug(f"Streamed tokens: {joined_buffer}")

                        await send(
                            {
                                "type": "http.response.body",
                                "body": joined_buffer.encode("utf-8"),
                                "more_body": True,
                            }
                        )
                        buffer = []

                if buffer and not timeout_reached:  # Don't send the last buffer of data if timeout.
                    joined_buffer = "".join(buffer)
                    await send(
                        {
                            "type": "http.response.body",
                            "body": joined_buffer.encode("utf-8"),
                            "more_body": False,
                        }
                    )

            except Exception as e:
                bt.logging.error(f"Error in forward: {e}")
                bt.logging.error(print_exception(type(e), e, e.__traceback__))
                if settings.NEURON_STOP_ON_FORWARD_EXCEPTION:
                    self.should_exit = True

            finally:
                synapse_latency = time.time() - init_time
                if settings.WANDB_ON:
                    self.log_event(
                        synapse=synapse,
                        timing=synapse_latency,
                        messages=messages,
                        accumulated_chunks=accumulated_chunks,
                        accumulated_chunks_timings=accumulated_chunks_timings,
                    )

        bt.logging.debug(
            f"📧 Message received from {synapse.dendrite.hotkey}, IP: {synapse.dendrite.ip}; \nForwarding synapse: {synapse}"
        )

        init_time = time.time()
        timeout_threshold = synapse.timeout

        token_streamer = partial(
            _forward,
            self,
            synapse,
            init_time,
            timeout_threshold,
        )
        return synapse.create_streaming_response(token_streamer)


# This is the main function, which runs the miner.
if __name__ == "__main__":
    with OpenAIMiner() as miner:
        while True:
            miner.log_status()
            time.sleep(5)

            if miner.should_exit:
                bt.logging.warning("Ending miner...")
                break
