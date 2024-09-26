# ruff: noqa: E402

# This is an example miner that can respond to the inference task using a vllm model.
from prompting import settings

settings.settings = settings.load_settings(mode="validator")
settings = settings.settings
import time
from functools import partial
from loguru import logger
from pydantic import model_validator
from prompting.base.miner import BaseStreamMinerNeuron
from prompting.base.protocol import StreamPromptingSynapse
from vllm import LLM, SamplingParams
from starlette.types import Send
from prompting.utils.logging import ErrorLoggingEvent, log_event
from prompting.base.protocol import AvailabilitySynapse
from prompting.llms.utils import GPUInfo

NEURON_MAX_TOKENS: int = 256
NEURON_TEMPERATURE: float = 0.7
NEURON_TOP_K: int = 50
NEURON_TOP_P: float = 0.95
NEURON_STREAMING_BATCH_SIZE: int = 12
NEURON_STOP_ON_FORWARD_EXCEPTION: bool = False

SYSTEM_PROMPT = """You are a helpful agent that does its best to answer all questions!"""


class VLLMMiner(BaseStreamMinerNeuron):
    llm: LLM | None = None
    accumulated_total_tokens: int = 0
    accumulated_prompt_tokens: int = 0
    accumulated_completion_tokens: int = 0
    accumulated_total_cost: float = 0
    should_exit: bool = False

    @model_validator(mode="after")
    def init_vllm(self) -> "VLLMMiner":
        GPUInfo.log_gpu_info()
        logger.debug("Loading vLLM model...")
        self.llm = LLM(model=settings.MINER_LLM_MODEL, gpu_memory_utilization=0.3)
        logger.debug("vLLM model loaded.")
        GPUInfo.log_gpu_info()
        return self

    def forward(self, synapse: StreamPromptingSynapse) -> StreamPromptingSynapse:
        """The forward function generates text based on a prompt, model, and seed."""

        async def _forward(
            self: "VLLMMiner",
            synapse: StreamPromptingSynapse,
            init_time: float,
            timeout_threshold: float,
            send: Send,
        ):
            buffer = []
            accumulated_chunks = []
            accumulated_chunks_timings = []
            temp_completion = ""  # for wandb logging
            timeout_reached = False

            try:
                start_time = time.time()
                sampling_params = SamplingParams(
                    seed=synapse.seed,
                )

                stream_response = self.llm.generate(prompts=[synapse.messages[0]], sampling_params=sampling_params)

                for chunk in stream_response:
                    chunk_content = chunk.outputs[0].text

                    if not chunk_content:
                        logger.info("vLLM returned chunk content with None")
                        continue

                    accumulated_chunks.append(chunk_content)
                    accumulated_chunks_timings.append(time.time() - start_time)

                    buffer.append(chunk_content)

                    if time.time() - init_time > timeout_threshold:
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
                logger.exception(e)
                logger.error(f"Error in forward: {e}")
                log_event(ErrorLoggingEvent(error=str(e)))
                if NEURON_STOP_ON_FORWARD_EXCEPTION:
                    self.should_exit = True

            finally:
                synapse_latency = time.time() - init_time
                self.log_event(
                    synapse=synapse,
                    timing=synapse_latency,
                    messages=synapse.messages,
                    accumulated_chunks=accumulated_chunks,
                    accumulated_chunks_timings=accumulated_chunks_timings,
                )

        logger.debug(
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

    def check_availability(self, synapse: AvailabilitySynapse) -> AvailabilitySynapse:
        """The check_availability function returns an AvailabilitySynapse which indicates
        which tasks and models this miner can handle."""

        logger.info(f"Checking availability of miner... {synapse}")
        synapse.task_availabilities = {
            task: True
            for task, _ in synapse.task_availabilities.items()
            if task == "SyntheticInferenceTask" or "OrganicInferenceTask"
        }
        synapse.llm_model_availabilities = {
            model: True for model, _ in synapse.llm_model_availabilities.items() if model == settings.MINER_LLM_MODEL
        }
        return synapse


if __name__ == "__main__":
    with VLLMMiner() as miner:
        while not miner.should_exit:
            miner.log_status()
            time.sleep(5)
        logger.warning("Ending miner...")
