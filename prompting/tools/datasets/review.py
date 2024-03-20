import random
import functools

import bittensor as bt
from typing import Dict, Union, List, Tuple
from .base import TemplateDataset


class ReviewDataset(TemplateDataset):
    "Review dataset, which creates LLM prompts for writing reviews."

    SENTIMENTS = ["positive", "neutral", "negative"]
    # TODO: filter nonsense combinations of params

    query_template = "Create a {topic} review of a {title} in the style of {mood} person in a {subtopic} tone. The review must be of {sentiment} sentiment."
    params = dict(
        topic=[
            "short",
            "long",
            "medium length",
            "twitter",
            "amazon",
            "terribly written",
            "hilarious",
        ],
        mood=["angry", "sad", "amused", "bored", "indifferent", "shocked", "terse"],
        subtopic=[
            "casual",
            "basic",
            "silly",
            "random",
            "thoughtful",
            "serious",
            "rushed",
        ],
        title=[
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
