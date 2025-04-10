import random
from typing import Any, ClassVar, Iterator
from datasets import load_dataset
from datasets.dataset_dict import DatasetDict, IterableDatasetDict
from datasets.arrow_dataset import Dataset
from datasets.iterable_dataset import IterableDataset
from pydantic import ConfigDict, model_validator

from shared.base import BaseDataset, DatasetEntry

ALLOWED_FILE_ENDINGS = {
    "python": [".py"],
    "js": [".js", ".jsx", ".ts", ".tsx"],
}
MIN_FILE_SIZE = 100
MAX_FILE_SIZE = 100_000
MIN_INPUT_LINES = 10
OUTPUT_LINES = 10
MAX_LINES = 500
RETRIES = 50  # Increased retry limit
RANDOM_SKIP = 1_000


class HuggingFaceGithubDatasetEntry(DatasetEntry):
    github_url: str
    file_path: str
    file_content: str
    source: str | None = None


class HuggingFaceGithubDataset(BaseDataset):
    language: str = "python"
    dataset: ClassVar[DatasetDict | Dataset | IterableDatasetDict | IterableDataset | None] = None
    iterator: ClassVar[Iterator[Any] | None] = None

    model_config = ConfigDict(arbitrary_types_allowed=True)

    @model_validator(mode="after")
    def load_dataset(self) -> "HuggingFaceGithubDataset":
        if HuggingFaceGithubDataset.dataset is None or self.iterator is None:
            HuggingFaceGithubDataset.dataset = load_dataset(
                "macrocosm-os/code-parrot-github-code", streaming=True, split="train", trust_remote_code=True
            )
            HuggingFaceGithubDataset.iterator = iter(HuggingFaceGithubDataset.dataset.filter(self._filter_function))
        return self

    def _filter_function(self, example):
        return (
            any(example["path"].endswith(ending) for ending in ALLOWED_FILE_ENDINGS[self.language])
            and MIN_FILE_SIZE <= int(example["size"]) <= MAX_FILE_SIZE
            and len(example["content"].split("\n")) >= (MIN_INPUT_LINES + OUTPUT_LINES)
        )

    def _process_entry(self, entry: dict) -> HuggingFaceGithubDatasetEntry:
        file_content = "\n".join(entry["content"].split("\n")[:MAX_LINES])
        url = f"https://github.com/{entry['repo_name']}"
        return HuggingFaceGithubDatasetEntry(
            github_url=url, file_path=entry["path"], file_content=file_content, source=url
        )

    def get(self) -> HuggingFaceGithubDatasetEntry:
        return self.next()

    def next(self) -> HuggingFaceGithubDatasetEntry:
        for _ in range(random.randint(0, RANDOM_SKIP)):
            next(HuggingFaceGithubDataset.iterator)

        for _ in range(RETRIES):
            try:
                entry = next(HuggingFaceGithubDataset.iterator)
                return self._process_entry(entry)  # Throws failed to get a valid file after multiple attempts
            except StopIteration:
                self.reset()
        raise Exception("Failed to get a valid file after multiple attempts")

    def random(self) -> HuggingFaceGithubDatasetEntry:
        # Note: The dataset is streamed, so true random access is not possible.
        # This method will just return the next item, similar to `next()`.
        return self.next()

    def reset(self):
        HuggingFaceGithubDataset.iterator = iter(HuggingFaceGithubDataset.dataset.filter(self._filter_function))


if __name__ == "__main__":
    dataset = HuggingFaceGithubDataset().load_dataset()
    entry = dataset.next()
