# ruff: noqa: E402
from prompting import settings

settings.settings = settings.Settings(mode="miner")
settings = settings.settings
import time
from functools import partial
from openai import OpenAI
from loguru import logger
from pydantic import model_validator
from prompting.base.miner import BaseStreamMinerNeuron
from prompting.base.protocol import StreamPromptingSynapse
from neurons.miners.openai.utils import OpenAIUtils
from starlette.types import Send
from prompting.utils.logging import ErrorEvent, log_event

MODEL_ID: str = "gpt-3.5-turbo"
NEURON_MAX_TOKENS: int = 256
NEURON_TEMPERATURE: float = 0.7
NEURON_TOP_K: int = 50
NEURON_TOP_P: float = 0.95
NEURON_STREAMING_BATCH_SIZE: int = 12
NEURON_STOP_ON_FORWARD_EXCEPTION: bool = False

SYSTEM_PROMPT = """You are a helpful agent that does it's best to answer all questions!"""


class OpenAIMiner(BaseStreamMinerNeuron, OpenAIUtils):
    """Langchain-based miner using OpenAI's API as the LLM.
    This miner relies entirely on the models' own representation and world model.
    """

    model: OpenAI | None = None
    accumulated_total_tokens: int = 0
    accumulated_prompt_tokens: int = 0
    accumulated_completion_tokens: int = 0
    accumulated_total_cost: float = 0
    should_exit: bool = False

    @model_validator(mode="after")
    def init_openai(self) -> "OpenAIMiner":
        self.model = OpenAI(api_key=settings.OPENAI_API_KEY)
        return self

    def forward(self, synapse: StreamPromptingSynapse) -> StreamPromptingSynapse:
        async def _forward(
            self: "OpenAIMiner",
            synapse: StreamPromptingSynapse,
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
                system_prompt_message = [{"role": "system", "content": SYSTEM_PROMPT}]
                synapse_messages = [
                    {"role": role, "content": message} for role, message in zip(synapse.roles, synapse.messages)
                ]

                messages = system_prompt_message + synapse_messages

                start_time = time.time()
                stream_response = self.model.chat.completions.create(
                    model=MODEL_ID,
                    messages=messages,
                    max_tokens=NEURON_MAX_TOKENS,
                    stream=True,
                )

                for chunk in stream_response:
                    chunk_content = chunk.choices[0].delta.content

                    if chunk_content is None:
                        logger.info("OpenAI returned chunk content with None")
                        continue

                    accumulated_chunks.append(chunk_content)
                    accumulated_chunks_timings.append(time.time() - start_time)

                    buffer.append(chunk_content)

                    if time.time() - start_time > timeout_threshold:
                        logger.debug("⏰ Timeout reached, stopping streaming")
                        timeout_reached = True
                        break

                    if len(buffer) == NEURON_STREAMING_BATCH_SIZE:
                        joined_buffer = "".join(buffer)
                        temp_completion += joined_buffer
                        logger.debug(f"Streamed tokens: {joined_buffer}")

                        await send(
                            {
                                "type": "http.response.body",
                                "body": joined_buffer.encode("utf-8"),
                                "more_body": True,
                            }
                        )
                        buffer = []

                if not buffer or timeout_reached:
                    return

                joined_buffer = "".join(buffer)
                await send(
                    {
                        "type": "http.response.body",
                        "body": joined_buffer.encode("utf-8"),
                        "more_body": False,
                    }
                )

            except Exception as e:
                logger.exception(e)
                logger.error(f"Error in forward: {e}")
                log_event(ErrorEvent(error=str(e)))
                if NEURON_STOP_ON_FORWARD_EXCEPTION:
                    self.should_exit = True

            finally:
                synapse_latency = time.time() - start_time
                self.log_event(
                    synapse=synapse,
                    timing=synapse_latency,
                    messages=messages,
                    accumulated_chunks=accumulated_chunks,
                    accumulated_chunks_timings=accumulated_chunks_timings,
                )

        logger.debug(
            f"📧 Message received from {synapse.dendrite.hotkey}, IP: {synapse.dendrite.ip}; \nForwarding synapse: {synapse}"
        )

        timeout_threshold = synapse.timeout

        token_streamer = partial(
            _forward,
            self,
            synapse,
            timeout_threshold,
        )

        streaming_response = synapse.create_streaming_response(token_streamer)
        return streaming_response


if __name__ == "__main__":
    with OpenAIMiner() as miner:
        while not miner.should_exit:
            miner.log_status()
            time.sleep(5)
        logger.warning("Ending miner...")
