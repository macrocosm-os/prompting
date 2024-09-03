from abc import ABC, abstractmethod
from pydantic import BaseModel
from typing import ClassVar
import json


class DatasetEntry(BaseModel):
    @property
    def hash(self) -> int:
        return hash(json.dumps(self.model_dump(), sort_keys=True))

    def __hash__(self) -> int:
        return self.hash


class MMLUEntry(DatasetEntry):
    query: str
    subject: str
    choices: list[str]
    answer: str


class Context(DatasetEntry):
    title: str
    topic: str
    subtopic: str
    content: str
    internal_links: list[str]
    external_links: list[str]
    source: str
    tags: list[str] | None = None
    extra: dict | None = None  # additional non-essential information
    stats: dict | None = None  # retrieval stats such as fetch time, number of tries, etc.


class BaseDataset(ABC, BaseModel):
    """Base class for datasets."""

    name: ClassVar[str] = "base"
    max_tries: int = 10

    @abstractmethod
    def random(self) -> DatasetEntry: ...

    def get(self) -> DatasetEntry:
        return self.random()

    def next(self) -> DatasetEntry:
        return self.random()
