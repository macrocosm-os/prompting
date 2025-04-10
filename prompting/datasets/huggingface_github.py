import random
from loguru import logger
from typing import Any, ClassVar, Iterator
from datasets import load_dataset
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
DEFAULT_NUM_SHARDS = 1126
RANDOM_SKIP = 100


class HuggingFaceGithubDatasetEntry(DatasetEntry):
    github_url: str
    file_path: str
    file_content: str
    source: str | None = None


class HuggingFaceGithubDataset(BaseDataset):
    language: str = "python"
    
    # Shared streaming dataset and its number of shards
    base_dataset: ClassVar[Any] = None
    num_shards: ClassVar[int] = DEFAULT_NUM_SHARDS

    # Instance-level iterator over the current shard
    current_shard_iterator: Iterator[Any] | None = None

    model_config = ConfigDict(arbitrary_types_allowed=True)

    @model_validator(mode="after")
    def load_dataset(self) -> "HuggingFaceGithubDataset":
        if HuggingFaceGithubDataset.base_dataset is None:
            # Load the full streaming dataset
            HuggingFaceGithubDataset.base_dataset = load_dataset(
                "macrocosm-os/code-parrot-github-code",
                streaming=True,
                split="train",
                trust_remote_code=True
            )
            # Try to determine the number of shards from the underlying file list.
            files = HuggingFaceGithubDataset.base_dataset._ex_iterable.kwargs.get("files")
            if files is not None:
                HuggingFaceGithubDataset.num_shards = len(files)
            else:
                print("Cannot get number of shards")
                HuggingFaceGithubDataset.num_shards = DEFAULT_NUM_SHARDS

        # Select a random shard to begin iterating
        self._reset_shard()
        return self

    def _reset_shard(self) -> None:
        """Choose a new random shard and creates a fresh iterator over its filtered records."""
        random_shard_index = random.randrange(HuggingFaceGithubDataset.num_shards)
        print(f"Shard index {random_shard_index}")
        shard_dataset = HuggingFaceGithubDataset.base_dataset.shard(
            num_shards=HuggingFaceGithubDataset.num_shards,
            index=random_shard_index
        )
        # Apply filtering to the selected shard
        shard_dataset = shard_dataset.filter(self._filter_function)
        HuggingFaceGithubDataset.current_shard_iterator = iter(shard_dataset)

    def _filter_function(self, example: dict) -> bool:
        return (
            any(example["path"].endswith(ending) for ending in ALLOWED_FILE_ENDINGS[self.language])
            and MIN_FILE_SIZE <= int(example["size"]) <= MAX_FILE_SIZE
            and len(example["content"].split("\n")) >= (MIN_INPUT_LINES + OUTPUT_LINES)
        )

    def _process_entry(self, entry: dict) -> HuggingFaceGithubDatasetEntry:
        file_content = "\n".join(entry["content"].split("\n")[:MAX_LINES])
        url = f"https://github.com/{entry['repo_name']}"
        return HuggingFaceGithubDatasetEntry(
            github_url=url,
            file_path=entry["path"],
            file_content=file_content,
            source=url
        )

    def _try_sample(self) -> HuggingFaceGithubDatasetEntry:
        """Return the next record from the current shard.

        When the shard is exhausted, it automatically resets to a new random shard.
        """
        try:
            entry = next(HuggingFaceGithubDataset.current_shard_iterator)
        except StopIteration:
            self._reset_shard()
            entry = next(HuggingFaceGithubDataset.current_shard_iterator)
        return entry

    def next(self) -> HuggingFaceGithubDatasetEntry:
        for _ in range(random.randint(0, RANDOM_SKIP)):
            self._try_sample()

        while True:
            try:
                entry = self._try_sample()
                return self._process_entry(entry)
            except BaseException as e:
                logger.warning(f"Failed to sample from shard: {e}")

    def get(self) -> HuggingFaceGithubDatasetEntry:
        return self.next()

    def random(self) -> HuggingFaceGithubDatasetEntry:
        return self.next()


if __name__ == "__main__":
    dataset = HuggingFaceGithubDataset().load_dataset()
    entry = dataset.next()
