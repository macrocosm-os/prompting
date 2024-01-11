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
    async def __call__(self, synapse: bt.Synapse , timeout: float) -> List[MockSynapse]:
        synapse = MockSynapse(
            status_message="OK",
            status_code=200,
            completion="Hello, world!",
            time=0.01
        )

        return [synapse]

# response = {
#     'name': 'Prompting',
#     'timeout': 1.0,
#     'total_size': 4023,
#     'header_size': 0,
#     'dendrite': TerminalInfo(
#         status_code=408,
#         status_message='Timedout after 1.0 seconds.',
#         process_time=None,
#         ip='187.214.177.93',
#         port=None,
#         version=650,
#         nonce=7560082750,
#         uuid='c5dd01b0-b0d2-11ee-a56a-4284d8373118',
#         hotkey='5Ey8t8pBJSYqLYCzeC3HiPJu5DxzXy2Dzheaj29wRHvhjoai', signature='0x0cc5765bb00dc6ac051d5b7f131ea24298e7d1037017da6fc4e99c5b669dc22485ac0393de545ab1bf50d815d53aba24d3bc6c2f4b06987e9257af9feca8838b'
#         ),
#     'axon': TerminalInfo(
#         status_code=None,
#         status_message=None,
#         process_time=None,
#         ip='1.2.3.4',
#         port=8008,
#         version=None,
#         nonce=None,
#         uuid=None,
#         hotkey='miner-hotkey-13',
#         signature=None
#         ),
#     'computed_body_hash': '',
#     'required_hash_fields': ['messages'],
#     'roles': ['user'],
#     'messages': ['mock reply'],
#     'completion': ''
#     }

class DendriteResponseEvent:
    def __init__(self, responses: List[bt.Synapse], uids: torch.IntTensor):

        bt.logging.info(f"responses: {responses}")
        bt.logging.info(f'first miner response full object: {responses[1].__dict__}')
        self.uids = uids
        self.completions = [response.completion for response in responses]
        self.timings = [response.axon.process_time or 0 for response in responses]
        self.status_messages = [response.axon.status_message for response in responses]
        self.status_codes = [response.axon.status_code for response in responses]

    def as_dict(self):
        return {
            'uids': self.uids.tolist(),
            'completions': self.completions,
            'timings': self.timings,
            'status_messages': self.status_messages,
            'status_codes': self.status_codes
        }
        
    def __repr__(self):
        return f'DendriteResponseEvent(uids={self.uids}, completions={self.completions}, timings={self.timings}, status_messages={self.status_messages}, status_codes={self.status_codes})'


if __name__ == "__main__":
    d = MockDendrite()
    r = d()
    print(r)