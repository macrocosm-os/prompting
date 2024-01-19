import pytest

from prompting.tools import MockDataset, CodingDataset, WikiDataset, StackOverflowDataset, DateQADataset, MathDataset




DATASETS = [
    MockDataset,
    CodingDataset,
    WikiDataset,
    StackOverflowDataset,
    DateQADataset,
    MathDataset,
]


@pytest.mark.parametrize('dataset', DATASETS)
def test_create_task(dataset):
    data = dataset()
    assert data is not None


@pytest.mark.parametrize('dataset', DATASETS)
def test_create_task(dataset):
    data = dataset()
    assert data.next() is not None