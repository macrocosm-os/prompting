from typing import List
from dataclasses import dataclass


@dataclass
class Context:
    # TODO: Pydantic model
    title: str
    topic: str
    subtopic: str
    content: str
    internal_links: List[str]
    external_links: List[str]
    source: str
    tags: List[str] = None
    extra: dict = None  # additional non-essential information
    stats: dict = None  # retrieval stats such as fetch time, number of tries, etc.


@dataclass
class BatchContext:
    results: List[Context]
    stats: dict = None
