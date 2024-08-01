import time
from abc import ABC, abstractmethod
from typing import Dict, Literal
from pydantic import BaseModel
from typing import ClassVar


class Context(BaseModel):
    # TODO: Pydantic model
    title: str
    topic: str
    subtopic: str
    content: str
    internal_links: list[str]
    external_links: list[str]
    source: str
    tags: list[str] = None
    extra: dict = None  # additional non-essential information
    stats: dict = None  # retrieval stats such as fetch time, number of tries, etc.


class BaseDataset(ABC, BaseModel):
    """Base class for datasets."""

    name: ClassVar[str] = "base"
    max_tries: int = 10

    @abstractmethod
    def search(self, name) -> Context: ...

    @abstractmethod
    def random(self, name) -> Context: ...

    @abstractmethod
    def get(self, name) -> Context: ...

    def next(self, method: Literal["random", "search", "get"] = "random", **kwargs) -> Dict:
        tries = 1
        t0 = time.time()

        context: Context  # for some reason the ls doesn't understand it's of type Context without this
        while True:
            # TODO: Multithread the get method so that we don't have to suffer nonexistent pages
            if method == "random":
                context = self.random(**kwargs)
            elif method == "search":
                context = self.search(**kwargs)
            elif method == "get":
                context = self.get(**kwargs)

            if context:
                break

        context.source = self.__class__.__name__
        context.stats = {
            "fetch_time": time.time() - t0,
            "num_tries": tries,
            "fetch_method": method,
            "next_kwargs": kwargs,
        }
        return context
