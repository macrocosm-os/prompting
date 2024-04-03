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
import itertools
import mathgenerator
import bittensor as bt
from sympy.parsing.latex import parse_latex
from typing import Dict, Union, List, Tuple


from .base import Dataset
from ..selector import Selector


class MathDataset(Dataset):
    topics_list = mathgenerator.getGenList()

    def __init__(self, seed=None):
        self.seed = seed
        self.rng = random.Random(seed)

    def get(
        self,
        name: str,
        selector: Selector = None,
        include: List = None,
        exclude: List = None,
        **kwargs,
    ) -> Dict:
        """Get a math problem.

        Args:
            name (str): Name of math problem to generate.
            selector (Selector, optional): Selector instance to choose a specific problem. Defaults to None.
            include (List, optional): _description_. Defaults to None.
            exclude (List, optional): _description_. Defaults to None.

        Returns:
            Dict: _description_
        """
        bt.logging.info(f"Getting math problem {name!r}")
        info = mathgenerator.generate_context(name, **kwargs)
        if info["reward_type"] != "float" or info["topic"] == 'computer_science':
            return None

        math_words = [
            "math",
            "mathematics",
            "mathematical",
            "math problem",
            "math technique",
        ]
        external_links = []
        # construct external links from randomly shuffled trigrams containing 2 words from the problem and 1 random math word
        # binary_to_decimal -> ['binary to', 'to decimal']
        for bigram in itertools.combinations(info["forward_words"], 2):
            words = list(bigram) + [random.choice(math_words)]
            # shuffle the words e.g. ['binary', 'decimal', 'math problem'] -> 'decimal binary math problem'
            external_links.append(" ".join(random.sample(words, len(words))))

        return {
            "title": info["topic"],  # title of math problem
            "topic": info["topic"],  # title of problem topic
            "subtopic": info["subtopic"],  # title of problem subtopic
            "content": info["problem"],  # problem statement
            "internal_links": [info["topic"], info["subtopic"]],  # internal links
            "external_links": external_links,
            "tags": info["forward_words"],
            "source": "Mathgenerator",
            "extra": {"reward_type": info["reward_type"], "solution": info["solution"]},
        }

    def search(
        self, name, selector: Selector, include: List = None, exclude: List = None
    ) -> Dict:
        raise NotImplementedError(
            f"Search is not implemented for {self.__class__.__name__}"
        )

    def random(self, selector: Selector, **kwargs):
        """Create a random math problem."""
        return self.get(name=None, selector=selector, **kwargs)
