import random

from prompting import mutable_globals
from prompting.datasets.base import DatasetEntry
from prompting.datasets.base import BaseDataset
from typing import ClassVar

from prompting.tasks.base_task import BaseTextTask


class MixtureOfMinersEntry(DatasetEntry):
    completions: list[str]
    uids: list[str]
    organic: bool
    primary_task: BaseTextTask


class MixtureOfMinersDataset(BaseDataset):
    name: ClassVar[str] = "response_cache"
    exception: ValueError | None = None

    class Config:
        arbitrary_types_allowed = True

    def get(self) -> MixtureOfMinersEntry:
        return self.sample()

    def next(self) -> MixtureOfMinersEntry:
        return self.sample()

    def random(self) -> MixtureOfMinersEntry:
        return self.sample()

    def sample(self) -> MixtureOfMinersEntry:
        """Sample the data from miners' response cache history, raises ValueError if no history is available."""
        if not mutable_globals.task_responses:
            raise ValueError("No task responses available in the cache.")

        # TODO: Filter tasks.
        sample = random.sample(mutable_globals.task_responses)
        entry = MixtureOfMinersEntry(
            completions=sample.response.completions,
            uids=sample.response.stream_results_uids,
            organic=False,
            primary_task=sample.task,
        )
        return entry
