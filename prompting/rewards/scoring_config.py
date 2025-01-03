from dataclasses import dataclass

from prompting.tasks.base_task import BaseTextTask
from shared.base import DatasetEntry
from shared.dendrite import DendriteResponseEvent


@dataclass
class ScoringConfig:
    task: BaseTextTask
    response: DendriteResponseEvent
    dataset_entry: DatasetEntry
    block: int
    step: int
    task_id: str
