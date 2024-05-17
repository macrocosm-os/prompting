import torch
from typing import List
from dataclasses import dataclass
from prompting.protocol import StreamPromptingSynapse
from prompting.utils.misc import serialize_exception_to_string
 
 
@dataclass
class SynapseStreamResult:
    exception: BaseException = None
    uid: int = None
    accumulated_chunks: List[str] = None
    accumulated_chunks_timings: List[float] = None
    tokens_per_chunk: List[int] = None
    synapse: StreamPromptingSynapse = None


class DendriteResponseEvent:
    def __init__(
        self, stream_results: SynapseStreamResult, uids: torch.LongTensor, timeout: float
    ):
        self.uids = uids
        self.completions = []
        self.status_messages = []
        self.status_codes = []
        self.timings = []
        self.stream_results_uids = []
        self.stream_results_exceptions = []
        self.stream_results_all_chunks = []
        self.stream_results_all_chunks_timings = []
        self.stream_results_all_tokens_per_chunk = []
        
        for stream_result in stream_results:
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
                self.timings.append(timeout)
            else:
                self.timings.append(0)

            self.stream_results_uids.append(stream_result.uid)
            self.stream_results_exceptions.append(serialize_exception_to_string(stream_result.exception))
            self.stream_results_all_chunks.append(stream_result.accumulated_chunks)
            self.stream_results_all_chunks_timings.append(stream_result.accumulated_chunks_timings)
            self.stream_results_all_tokens_per_chunk.append(stream_result.tokens_per_chunk)

    def __state_dict__(self):
        return {
            "uids": self.uids.tolist(),
            "completions": self.completions,
            "timings": self.timings,
            "status_messages": self.status_messages,
            "status_codes": self.status_codes,
            "stream_results_uids": self.stream_results_uids,
            "stream_results_exceptions": self.stream_results_exceptions,
            "stream_results_all_chunks": self.stream_results_all_chunks,
            "stream_results_all_chunks_timings": self.stream_results_all_chunks_timings,
        }

    def __repr__(self):
        return f"DendriteResponseEvent(uids={self.uids}, completions={self.completions}, timings={self.timings}, status_messages={self.status_messages}, status_codes={self.status_codes})"
