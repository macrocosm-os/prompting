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
from abc import ABC, abstractmethod
from typing import Dict
import bittensor as bt

from ..selector import Selector
from .context import Context
from prompting.utils.exceptions import MaxRetryError


class Dataset(ABC):
    """Base class for datasets."""

    max_tries: int = 10

    @abstractmethod
    def search(self, name):
        ...

    @abstractmethod
    def random(self, name):
        ...

    @abstractmethod
    def get(self, name):
        ...

    def next(self, method: str = 'random', selector: Selector = Selector(), **kwargs) -> Dict:
        tries = 1
        t0 = time.time()

        while True:

            # TODO: Multithread the get method so that we don't have to suffer nonexistent pages
            info = {}
            if method == 'random':
                info = self.random(selector=selector, **kwargs)
            elif method == 'search':
                info = self.search(selector=selector, **kwargs)
            elif method == 'get':
                info = self.get(selector=selector, **kwargs)
            else:
                raise ValueError(f"Unknown dataset get method {method!r}")

            if info:
                break

            bt.logging.debug(f"Could not find any samples which meet {self.__class__.__name__} requirements after {tries} tries. Retrying... ({self.max_tries - tries} tries remaining.)")

            tries += 1
            if tries >= self.max_tries:
                raise MaxRetryError(
                    f"Could not find any samples which meet {self.__class__.__name__} requirements after {tries} tries."
                )

        info['stats'] = {
            'creator': self.__class__.__name__,
            'fetch_time': time.time() - t0,
            'num_tries': tries,
            'fetch_method': method,
            'next_kwargs': kwargs
            }
        return Context(**info)
