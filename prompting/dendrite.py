import torch
import bittensor as bt
from typing import List


class DendriteResponseEvent:
    def __init__(self, responses: List[bt.Synapse], uids: torch.LongTensor):
        bt.logging.info(f"responses: {responses}")

        self.uids = uids
        self.completions = [response.completion for response in responses]
        self.timings = [
            response.axon.process_time or 0 for response in responses
        ]
        self.status_messages = [
            response.axon.status_message for response in responses
        ]
        self.status_codes = [
            response.axon.status_code for response in responses
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

