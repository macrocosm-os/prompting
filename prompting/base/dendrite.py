from prompting.base.protocol import StreamPromptingSynapse
from prompting.utils.misc import serialize_exception_to_string
from pydantic import BaseModel, model_validator, ConfigDict


class SynapseStreamResult(BaseModel):
    exception: BaseException | None = None
    uid: int | None = None
    accumulated_chunks: list[str] | None = None
    accumulated_chunks_timings: list[float] | None = None
    tokens_per_chunk: list[int] | None = None
    synapse: StreamPromptingSynapse | None = None

    model_config = ConfigDict(arbitrary_types_allowed=True)

    def model_dump(self):
        # without a custom model dump, this leads to serialization errors in DendriteResponseEvent...
        return {
            "exception": self.exception,
            "uid": self.uid,
            "accumulated_chunks": self.accumulated_chunks,
            "accumulated_chunks_timings": self.accumulated_chunks_timings,
            "tokens_per_chunk": self.tokens_per_chunk,
            "synapse": self.synapse.model_dump(),
        }


class DendriteResponseEvent(BaseModel):
    uids: list[int]
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
        for stream_result in self.stream_results:
            # when passing this to a pydantic model, this method can be called multiple times, leading
            # to duplicating the arrays. If the arrays are already filled, we can skip this step
            if len(self.completions) > 0:
                return self
            # for some reason the language server needs this line to understand the type of stream_result
            stream_result: SynapseStreamResult

            synapse = stream_result.synapse

            self.completions.append(synapse.completion)
            self.status_messages.append(synapse.dendrite.status_message)
            status_code = synapse.dendrite.status_code

            if len(synapse.completion) == 0 and status_code == 200:
                status_code = 204

            self.status_codes.append(status_code)
            process_time = synapse.dendrite.process_time or 0
            if status_code == 200 or status_code == 204:
                self.timings.append(process_time)
            elif status_code == 408:
                self.timings.append(self.timeout)
            else:
                self.timings.append(0)

            self.stream_results_uids.append(stream_result.uid)
            self.stream_results_exceptions.append(serialize_exception_to_string(stream_result.exception))
            self.stream_results_all_chunks.append(stream_result.accumulated_chunks)
            self.stream_results_all_chunks_timings.append(stream_result.accumulated_chunks_timings)
            self.stream_results_all_tokens_per_chunk.append(stream_result.tokens_per_chunk)
        return self
