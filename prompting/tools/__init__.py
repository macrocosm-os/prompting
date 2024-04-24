from .datasets import (
    Context,
    Dataset,
    MockDataset,
    HFCodingDataset,
    WikiDataset,
    StackOverflowDataset,
    WikiDateDataset,
    MathDataset,
    GenericInstructionDataset,
)
from .selector import Selector

DATASETS = {
    # HFCodingDataset.name: HFCodingDataset,
    WikiDataset.name: WikiDataset,
    # StackOverflowDataset.name: StackOverflowDataset,
    MathDataset.name: MathDataset,
    WikiDateDataset.name: WikiDateDataset,
    GenericInstructionDataset.name: GenericInstructionDataset,
}
