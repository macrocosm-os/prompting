import numpy as np
import time
from prompting.base.protocol import StreamPromptingSynapse
from prompting.utils.misc import serialize_exception_to_string
from pydantic import BaseModel, model_validator
from loguru import logger
from typing import AsyncGenerator
from pydantic import ConfigDict


class SynapseStreamResult(BaseModel):
    exception: BaseException | None = None
    uid: int | None = None
    accumulated_chunks: list[str] | None = None
    accumulated_chunks_timings: list[float] | None = None
    tokens_per_chunk: list[int] | None = None
    synapse: StreamPromptingSynapse | None = None

    model_config = ConfigDict(arbitrary_types_allowed=True)

    @property
    def completion(self) -> str:
        if not self.synapse:
            logger.warning("Synapse is None")
            return
        return self.synapse.completion

    def model_dump(self):
        # without a custom model dump, this leads to serialization errors in DendriteResponseEvent...
        # TODO: This isn't great, ideally find a cleaner workaround
        return {
            "exception": self.exception,
            "uid": self.uid,
            "accumulated_chunks": self.accumulated_chunks,
            "accumulated_chunks_timings": self.accumulated_chunks_timings,
            "tokens_per_chunk": self.tokens_per_chunk,
            "synapse": self.synapse.model_dump() if self.synapse is not None else None,
        }


class StreamResultsParser:
    """Parser for streaming results that accumulates chunks into SynapseStreamResult objects"""

    def __init__(self):
        self.results_by_uid: dict[int, SynapseStreamResult] = {}

    def _ensure_result_exists(self, uid: int) -> None:
        """Ensure a result object exists for the given UID"""
        if uid not in self.results_by_uid:
            self.results_by_uid[uid] = SynapseStreamResult(
                uid=uid,
                accumulated_chunks=[],
                accumulated_chunks_timings=[],
                tokens_per_chunk=[],
            )

    async def parse_streaming_response(
        self, stream: AsyncGenerator[dict, None], synapse: StreamPromptingSynapse
    ) -> list[SynapseStreamResult]:
        """
        Parse a streaming response into a list of SynapseStreamResult objects

        Args:
            stream: The streaming response from send_streaming_request
            synapse_type: The type of synapse being used

        Returns:
            list of SynapseStreamResult objects, one per miner
        """
        try:
            async for chunk in stream:
                uid = chunk.get("uid")
                if uid is None:
                    logger.error(f"Received chunk without UID: {chunk}")
                    continue

                self._ensure_result_exists(uid)
                result = self.results_by_uid[uid]

                if "error" in chunk:
                    # Handle error case
                    result.exception = Exception(chunk["error"])
                    continue

                # Process successful chunk
                try:
                    chunk_data = chunk["data"]
                    result.accumulated_chunks.append(chunk_data)
                    result.accumulated_chunks_timings.append(time.time())

                    # You might want to implement token counting based on your tokenizer
                    # This is a placeholder that counts characters
                    result.tokens_per_chunk.append(len(chunk_data))

                except Exception as e:
                    result.exception = e
                    logger.error(f"Error processing chunk for UID {uid}: {e}")

            # After all chunks are processed, create final synapses
            for result in self.results_by_uid.values():
                if not result.exception:
                    try:
                        # Combine all chunks into final completion
                        combined_text = "".join(result.accumulated_chunks)
                        synapse.completion = combined_text
                        result.synapse = synapse
                    except Exception as e:
                        result.exception = e
                        logger.error(f"Error creating final synapse for UID {result.uid}: {e}")

        except Exception as e:
            logger.error(f"Error in stream parsing: {e}")
            # Create error result for all miners if we can't process the stream
            for uid in self.results_by_uid:
                self.results_by_uid[uid].exception = e

        return list(self.results_by_uid.values())


class DendriteResponseEvent(BaseModel):
    uids: np.ndarray | list[float]
    timeout: float
    stream_results: list[SynapseStreamResult]
    completions: list[str] = []
    status_messages: list[str] = []
    status_codes: list[int] = []
    timings: list[float] = []
    stream_results_uids: list[int] = []
    stream_results_exceptions: list[str] = []
    stream_results_all_chunks: list[list[str]] = []
    stream_results_all_chunks_timings: list[list[float]] = []
    stream_results_all_tokens_per_chunk: list[list[int]] = []

    model_config = ConfigDict(arbitrary_types_allowed=True)

    @model_validator(mode="after")
    def process_stream_results(self) -> "DendriteResponseEvent":
        # when passing this to a pydantic model, this method can be called multiple times, leading
        # to duplicating the arrays. If the arrays are already filled, we can skip this step
        if len(self.completions) > 0:
            return self
        for stream_result in self.stream_results:
            # for some reason the language server needs this line to understand the type of stream_result
            stream_result: SynapseStreamResult

            synapse = stream_result.synapse

            self.completions.append(synapse.completion)
            # self.status_messages.append(synapse.dendrite.status_message)
            # status_code = synapse.dendrite.status_code

            # if len(synapse.completion) == 0 and status_code == 200:
            #     status_code = 204

            # self.status_codes.append(status_code)
            # process_time = synapse.dendrite.process_time or 0
            # if status_code == 200 or status_code == 204:
            # self.timings.append(process_time)
            # elif status_code == 408:
            #     self.timings.append(self.timeout)
            # else:
            self.timings.append(0)

            self.stream_results_uids.append(stream_result.uid)
            self.stream_results_exceptions.append(serialize_exception_to_string(stream_result.exception))
            self.stream_results_all_chunks.append(stream_result.accumulated_chunks)
            self.stream_results_all_chunks_timings.append(stream_result.accumulated_chunks_timings)
        return self
