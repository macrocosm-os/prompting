# The MIT License (MIT)
# Copyright © 2024 Yuma Rao
# Copyright © 2023 Opentensor Foundation

# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
# documentation files (the “Software”), to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software,
# and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all copies or substantial portions of
# the Software.

# THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO
# THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
# OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.

import time
import random
import functools
from abc import ABC, abstractmethod
from typing import Dict
import bittensor as bt

from ..selector import Selector
from .context import Context, BatchContext
from prompting.utils.exceptions import MaxRetryError


class Dataset(ABC):
    """Base class for datasets."""

    max_tries: int = 10

    @abstractmethod
    def search(self, name): ...

    @abstractmethod
    def random(self, name): ...

    @abstractmethod
    def get(self, name): ...

    def next(
        self, method: str = "random", selector: Selector = Selector(), **kwargs
    ) -> Dict:
        tries = 1
        t0 = time.time()

        while True:
            # TODO: Multithread the get method so that we don't have to suffer nonexistent pages
            info = {}
            if method == "random":
                info = self.random(selector=selector, **kwargs)
            elif method == "search":
                info = self.search(selector=selector, **kwargs)
            elif method == "get":
                info = self.get(selector=selector, **kwargs)
            else:
                raise ValueError(f"Unknown dataset get method {method!r}")

            if info:
                break

            bt.logging.debug(
                f"Could not find any samples which meet {self.__class__.__name__} requirements after {tries} tries. Retrying... ({self.max_tries - tries} tries remaining.)"
            )

            tries += 1
            if tries >= self.max_tries:
                raise MaxRetryError(
                    f"Could not find any samples which meet {self.__class__.__name__} requirements after {tries} tries."
                )

        info["source"] = self.__class__.__name__
        info["stats"] = {
            "fetch_time": time.time() - t0,
            "num_tries": tries,
            "fetch_method": method,
            "next_kwargs": kwargs,
        }
        return Context(**info)


class BatchDataset(ABC):
    """Base class for batch datasets."""

    max_tries: int = 10
    batch_size: int = 16  # ensure that child classes contain batch_size attrib

    @abstractmethod
    async def random(self, name): ...

    async def next(
        self, method: str = "random", selector: Selector = Selector(), **kwargs
    ) -> BatchContext:
        t0 = time.time()

        for tries in range(1, self.max_tries + 1):
            if method == "random":
                results = await self.random()
                stats = {
                    "creator": self.__class__.__name__,
                    "fetch_time": time.time() - t0,
                    "num_tries": tries,
                    "fetch_method": method,
                    "next_kwargs": kwargs,
                }

                return BatchContext(results=results, stats=stats)
            else:
                raise ValueError(f"Unknown dataset get method {method!r}")

        # If no valid info is found after max_tries
        raise MaxRetryError(
            f"Could not find any samples which meet {self.__class__.__name__} requirements after {self.max_tries} tries."
        )


class TemplateDataset(Dataset):
    """Base class for datasets based on a template."""

    @property
    def size(self):
        return functools.reduce(
            lambda x, y: x * y, [len(v) for v in self.params.values()], 1
        )

    def __repr__(self):
        return f"{self.__class__.__name__} with template: {self.query_template!r} and {self.size} possible phrases"

    def get(self, params: dict):
        content = self.query_template.format(**params)
        keys, values = list(zip(*params.items()))

        return {
            "title": params.get(
                "title", keys[0]
            ),  # Use the first key as the title if no field called title is present
            "topic": params.get("topic", keys[min(1, len(keys) - 1)]),  # Same for topic
            "subtopic": params.get(
                "subtopic", keys[min(2, len(keys) - 2)]
            ),  # Same for subtopic
            "content": content,  # content
            "internal_links": values,  # internal links
            "external_links": values,  # external links
            "tags": values,  # tags
            "extra": {},
        }

    def random(self, selector: Selector = None):
        selected = {k: selector(v) for k, v in self.params.items()}
        return self.get(selected)

    def search(self, params: dict, selector: Selector = None):
        selected = {k: params.get(k, selector(v)) for k, v in self.params.items()}
        return self.get(selected)
