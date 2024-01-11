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
    async def __call__(self, synapse: bt.Synapse , timeout: float) -> List[MockSynapse]:                
        synapse = MockSynapse(
            status_message="OK",
            status_code=200,
            completion="Hello, world!",
            time=0.01
        )

        return [synapse]

if __name__ == "__main__":    
    d = MockDendrite()
    r = d()
    print(r)