from .datasets import (
    WikiDataset,
    WikiDateDataset,
)

DATASETS = {
    # HFCodingDataset.name: HFCodingDataset,
    WikiDataset.name: WikiDataset,
    # StackOverflowDataset.name: StackOverflowDataset,
    WikiDateDataset.name: WikiDateDataset,
}
