from prompting.datasets.base import BaseDataset
from prompting.datasets.base import DatasetEntry


class SyntheticInferenceDataset(BaseDataset):
    def search(self, name) -> DatasetEntry:
        return DatasetEntry()

    def random(self) -> DatasetEntry:
        return DatasetEntry()

    def get(self) -> DatasetEntry:
        return DatasetEntry()

    def next(self) -> DatasetEntry:
        return DatasetEntry()
