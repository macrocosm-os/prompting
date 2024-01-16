import torch
import bittensor as bt
from typing import List


class DendriteResponseEvent:
    def __init__(self, responses: List[bt.Synapse], uids: torch.LongTensor):

        self.uids = uids
        self.completions = [synapse.completion for synapse in responses]
        self.timings = [
            synapse.dendrite.process_time or 0 for synapse in responses
        ]
        self.status_messages = [
            synapse.dendrite.status_message for synapse in responses
        ]
        self.status_codes = [
            synapse.dendrite.status_code for synapse in responses
        ]

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

