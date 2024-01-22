import pytest

from .fixtures.dataset import DATASETS, CONTEXTS, CONTEXT_FIELDS


@pytest.mark.parametrize('dataset', DATASETS)
def test_create_dataset(dataset):
    data = dataset()
    assert data is not None


@pytest.mark.parametrize('dataset', DATASETS)
def test_context_is_dict(dataset):
    assert type(CONTEXTS[dataset]) == dict

@pytest.mark.parametrize('dataset', DATASETS)
def test_dataset_context_contains_expected_fields(dataset):
    assert set(CONTEXTS[dataset].keys()) == CONTEXT_FIELDS[dataset]


