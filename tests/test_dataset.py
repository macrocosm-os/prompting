import pytest

from .fixtures.dataset import DATASETS, CONTEXTS, CONTEXT_FIELDS
from prompting.tools.datasets import Dataset
from prompting.tools import Context


@pytest.mark.parametrize('dataset', DATASETS)
def test_create_dataset(dataset: Dataset):
    ds = dataset()
    assert ds is not None


@pytest.mark.parametrize('dataset', DATASETS)
def test_dataset_is_subclass_of_dataset_class(dataset: Dataset):
    ds = dataset()
    assert issubclass(type(ds), Dataset)


@pytest.mark.parametrize('dataset', DATASETS)
@pytest.mark.parametrize('method', ('next', 'get', 'random', 'search'))
def test_dataset_has_expected_methods(dataset: Dataset, method: str):
    ds = dataset()
    assert hasattr(ds, method)
    assert callable(getattr(ds, method))


@pytest.mark.skip(reason="Not implemented")
@pytest.mark.parametrize('dataset', DATASETS)
@pytest.mark.parametrize('method', ('next', 'get', 'random', 'search'))
def test_dataset_methods_return_contexts(dataset: Dataset, method: str):
    ds = dataset()
    assert hasattr(ds, method)
    assert callable(getattr(ds, method))


@pytest.mark.parametrize('dataset', DATASETS)
def test_context_is_of_type_context_class(dataset: Dataset):
    assert type(CONTEXTS[dataset]) == Context


@pytest.mark.parametrize('dataset', DATASETS)
@pytest.mark.parametrize('field', CONTEXT_FIELDS.keys())
def test_context_contains_expected_field(dataset: Dataset, field: str):
    assert hasattr(CONTEXTS[dataset], field)


@pytest.mark.parametrize('dataset', DATASETS)
@pytest.mark.parametrize('field, expected_type', list(CONTEXT_FIELDS.items()))
def test_context_field_has_expected_types(dataset: Dataset, field: str, expected_type: type):
    assert isinstance(getattr(CONTEXTS[dataset], field), expected_type)


@pytest.mark.parametrize('dataset', DATASETS)
@pytest.mark.parametrize('field', CONTEXT_FIELDS.keys())
def test_context_field_is_not_null(dataset: Dataset, field: str):
    assert getattr(CONTEXTS[dataset], field)


@pytest.mark.parametrize('dataset', DATASETS)
@pytest.mark.parametrize('field', ('creator', 'fetch_time', 'num_tries', 'fetch_method', 'next_kwargs'))
def test_context_stats_field_contains_expected_keys(dataset: Dataset, field: str):
    assert field in CONTEXTS[dataset].stats