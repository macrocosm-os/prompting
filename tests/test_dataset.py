import pytest

from fixtures.llm import LLM_PIPELINE
from fixtures.tasks import CONTEXTS, TASKS
from fixtures.dataset import DATASETS


@pytest.mark.parametrize('dataset', DATASETS)
def test_create_task(dataset):
    data = dataset()
    assert data is not None


@pytest.mark.parametrize('dataset', CONTEXTS)
def test_context_fields(context):
    assert context is not None
    