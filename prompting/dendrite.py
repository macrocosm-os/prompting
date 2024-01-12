import torch
import bittensor as bt
from typing import List
from dataclasses import dataclass


@dataclass
class MockSynapse:
    status_message: str
    status_code: int
    completion: str
    time: float


class MockDendrite:
    async def __call__(
        self, synapse: bt.Synapse, timeout: float
    ) -> List[MockSynapse]:
        synapse = MockSynapse(
            status_message="OK",
            status_code=200,
            completion="Hello, world!",
            time=0.01,
        )

        return [synapse]

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

    def as_dict(self):
        return {
            "uids": self.uids.tolist(),
            "completions": self.completions,
            "timings": self.timings,
            "status_messages": self.status_messages,
            "status_codes": self.status_codes,
        }

    def __repr__(self):
        return f"DendriteResponseEvent(uids={self.uids}, completions={self.completions}, timings={self.timings}, status_messages={self.status_messages}, status_codes={self.status_codes})"


if __name__ == "__main__":
    d = MockDendrite()
    r = d()
    print(r)
