from .datasets import (
    Dataset,
    MockDataset,
    OrganicDataset,
    HFCodingDataset,
    WikiDataset,
    StackOverflowDataset,
    WikiDateDataset,
    MathDataset,
    GenericInstructionDataset,
    ReviewDataset,
)
from .selector import Selector

DATASETS = {
    #HFCodingDataset.name: HFCodingDataset,
    OrganicDataset.name: OrganicDataset,
    WikiDataset.name: WikiDataset,
    #StackOverflowDataset.name: StackOverflowDataset,
    MathDataset.name: MathDataset,
    WikiDateDataset.name: WikiDateDataset,
    GenericInstructionDataset.name: GenericInstructionDataset, 
    ReviewDataset.name: ReviewDataset
}
