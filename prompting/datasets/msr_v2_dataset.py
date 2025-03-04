from shared.base import BaseDataset, Context, DatasetEntry
from typing import ClassVar
import random

dataset_entry_queue: list[Context] = []

class MSRDiscriminatorDatasetEntry(DatasetEntry):
    miner_response: str
    validator_reference: str
    miner_uid: int

class MSRDiscriminatorDataset(BaseDataset):
    name: ClassVar[str] = "msr_discriminator"

    def random(self) -> Context:
        return random.choice(dataset_entry_queue)
    
    @classmethod
    def add_entry(cls, miner_response: str, validator_reference: str, miner_uid: int):
        dataset_entry_queue.append(MSRDiscriminatorDatasetEntry(
            miner_response=miner_response,
            validator_reference=validator_reference,
            miner_uid=miner_uid
        ))

    @classmethod
    def get_entry(cls, miner_uid: int) -> MSRDiscriminatorDatasetEntry:
        return next((entry for entry in dataset_entry_queue if entry.miner_uid == miner_uid), None)
    
