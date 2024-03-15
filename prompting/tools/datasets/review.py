import random
import functools

import bittensor as bt
from typing import Dict, Union, List, Tuple
from .base import Dataset


class ReviewDataset(Dataset):
    SENTIMENTS = ["positive", "neutral", "negative"]
    # TODO: filter nonsense combinations of params

    query_template = "Create a {style} review of a {topic} in the style of {mood} person in a {tone} tone. The review must be of {sentiment} sentiment."
    params = dict(
        style=[
            "short",
            "long",
            "medium length",
            "twitter",
            "amazon",
            "terribly written",
            "hilarious",
        ],
        mood=["angry", "sad", "amused", "bored", "indifferent", "shocked", "terse"],
        tone=["casual", "basic", "silly", "random", "thoughtful", "serious", "rushed"],
        topic=[
            "movie",
            "book",
            "restaurant",
            "hotel",
            "product",
            "service",
            "car",
            "company",
            "live event",
        ],
        sentiment=SENTIMENTS,
    )

    @property
    def size(self):
        return functools.reduce(
            lambda x, y: x * y, [len(v) for v in self.params.values()], 1
        )

    def __repr__(self):
        return f"{self.__class__.__name__} with template: {self.query_template!r} and {self.size} possible phrases"

    def random(self, *args, **kwargs):
        selected = {k: random.choice(v) for k, v in self.params.items()}
        links_unused = list(selected.values())
        return {
            "title": f'A {selected["sentiment"]} review of a {selected["topic"]}',
            "topic": selected["topic"],
            "subtopic": selected["sentiment"],
            "content": self.query_template.format(**selected),
            "internal_links": links_unused,
            "external_links": links_unused,
            "source": self.__class__.__name__,
        }

    def search(self, *args, **kwargs):
        return self.random()

    def get(self, *args, **kwargs):
        return self.random()
