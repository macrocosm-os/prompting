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
        
        synapses = [stream_result.synapse for stream_result in stream_results]

        for synapse in synapses:
            self.completions.append(synapse.completion)
            self.status_messages.append(synapse.dendrite.status_message)

            if len(synapse.completion) == 0 and synapse.dendrite.status_code == 200:
                synapse.dendrite.status_code = 204

            self.status_codes.append(synapse.dendrite.status_code)

            if (synapse.dendrite.process_time) and (
                synapse.dendrite.status_code == 200
                or synapse.dendrite.status_code == 204
            ):
                self.timings.append(synapse.dendrite.process_time)
            elif synapse.dendrite.status_code == 408:
                self.timings.append(timeout)
            else:
                self.timings.append(0)  # situation where miner is not alive

        self.completions = [synapse.completion for synapse in synapses]
        self.timings = [
            synapse.dendrite.process_time or timeout for synapse in synapses
        ]
        self.status_messages = [
            synapse.dendrite.status_message for synapse in synapses
        ]
        self.status_codes = [synapse.dendrite.status_code for synapse in synapses]
        
        self.stream_results_uids = [stream_result.uid for stream_result in stream_results]
        self.stream_results_exceptions = [
            serialize_exception_to_string(stream_result.exception)
            for stream_result in stream_results
        ]
        self.stream_results_all_chunks = [stream_result.accumulated_chunks for stream_result in stream_results]
        self.stream_results_all_chunks_timings = [stream_result.accumulated_chunks_timings for stream_result in stream_results]

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
