import pytest
from .fixtures.dataset import DATASETS, CONTEXTS, CONTEXT_FIELDS, BATCH_DATASETS
from prompting.tools.datasets import Dataset
from prompting.tools import Context


@pytest.mark.parametrize("dataset", DATASETS)
def test_create_dataset(dataset: Dataset):
    ds = dataset()
    assert ds is not None


@pytest.mark.parametrize("dataset", DATASETS)
def test_dataset_is_subclass_of_dataset_class(dataset: Dataset):
    ds = dataset()
    assert issubclass(type(ds), Dataset)


@pytest.mark.parametrize("dataset", DATASETS)
@pytest.mark.parametrize("method", ("next", "get", "random", "search"))
def test_dataset_has_expected_methods(dataset: Dataset, method: str):
    ds = dataset()
    assert hasattr(ds, method)
    assert callable(getattr(ds, method))


@pytest.mark.skip(reason="Not implemented")
@pytest.mark.parametrize("dataset", DATASETS)
@pytest.mark.parametrize("method", ("next", "get", "random", "search"))
def test_dataset_methods_return_contexts(dataset: Dataset, method: str):
    ds = dataset()
    assert hasattr(ds, method)
    assert callable(getattr(ds, method))


@pytest.mark.parametrize("dataset", DATASETS)
def test_context_is_of_type_context_class(dataset: Dataset):
    assert type(CONTEXTS[dataset]) == Context


@pytest.mark.parametrize("dataset", DATASETS)
@pytest.mark.parametrize("field", CONTEXT_FIELDS.keys())
def test_context_contains_expected_field(dataset: Dataset, field: str):
    assert hasattr(CONTEXTS[dataset], field)


@pytest.mark.parametrize("dataset", DATASETS)
@pytest.mark.parametrize("field, expected_type", list(CONTEXT_FIELDS.items()))
def test_context_field_has_expected_types(
    dataset: Dataset, field: str, expected_type: type
):
    assert isinstance(getattr(CONTEXTS[dataset], field), expected_type)


@pytest.mark.parametrize("dataset", DATASETS)
@pytest.mark.parametrize("field", CONTEXT_FIELDS.keys())
def test_context_field_is_not_null(dataset: Dataset, field: str):
    assert getattr(CONTEXTS[dataset], field)


@pytest.mark.parametrize("dataset", DATASETS)
@pytest.mark.parametrize(
    "field", ("fetch_time", "num_tries", "fetch_method", "next_kwargs")
)
def test_context_stats_field_contains_expected_keys(dataset: Dataset, field: str):
    assert field in CONTEXTS[dataset].stats


## Batch dataset tests
@pytest.mark.asyncio
@pytest.mark.parametrize("dataset", BATCH_DATASETS)
@pytest.mark.parametrize("batch_size", [1, 2, 4, 8])
async def test_batch_size_parameter(dataset, batch_size):
    batch_context = await dataset(batch_size=batch_size).next()
    results = batch_context.results
    # Check if results match expected batch size
    assert len(results) == batch_size
    assert type(results) == list
    assert all(type(result) == Context for result in results)


@pytest.mark.asyncio
@pytest.mark.parametrize("dataset", BATCH_DATASETS)
async def test_random_batch_retrieval(dataset):
    # Fetch batches
    batch1_results = (await dataset(batch_size=2).next()).results
    batch2_results = (await dataset(batch_size=2).next()).results

    # Check that batches have different elements
    assert batch1_results != batch2_results

    # Check that results are of expected type
    assert type(batch1_results) == list
    assert all(type(result) == Context for result in batch1_results)

    assert type(batch2_results) == list
    assert all(type(result) == Context for result in batch2_results)
