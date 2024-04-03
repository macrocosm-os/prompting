import torch
import bittensor as bt
from typing import List


class DendriteResponseEvent:
    def __init__(
        self, responses: List[bt.Synapse], uids: torch.LongTensor, timeout: float
    ):
        self.uids = uids
        self.completions = []
        self.status_messages = []
        self.status_codes = []
        self.timings = []

        for synapse in responses:
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

    def __state_dict__(self):
        return {
            "uids": self.uids.tolist(),
            "completions": self.completions,
            "timings": self.timings,
            "status_messages": self.status_messages,
            "status_codes": self.status_codes,
        }

    def __repr__(self):
        return f"DendriteResponseEvent(uids={self.uids}, completions={self.completions}, timings={self.timings}, status_messages={self.status_messages}, status_codes={self.status_codes})"
