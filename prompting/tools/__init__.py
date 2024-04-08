from .datasets import (
    Context,
    Dataset,
    MockDataset,
    HFCodingDataset,
    WikiDataset,
    StackOverflowDataset,
    WikiDateDataset,
    MathDataset,
)
from .selector import Selector

DATASETS = {
    "mock": MockDataset,
    "hf_coding": HFCodingDataset,
    "wiki": WikiDataset,
    #"stack_overflow": StackOverflowDataset,
    "wiki_date": WikiDateDataset,
    "math": MathDataset,
}
