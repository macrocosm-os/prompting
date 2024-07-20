import random
import functools

import bittensor as bt
from typing import Dict, Union, List, Tuple
from .base import TemplateDataset


class ReviewDataset(TemplateDataset):
    "Review dataset, which creates LLM prompts for writing reviews."
    name : str = 'review'
    SENTIMENTS : List = ["positive", "neutral", "negative"]
    # TODO: Expand the params to create a larger dataset, while maintaining the same quality. 
    query_template : str = "Create a {topic} review of a {title}. The review must be of {subtopic} sentiment."
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
        subtopic=SENTIMENTS,
    )
    