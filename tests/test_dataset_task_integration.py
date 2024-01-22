import pytest
from prompting.tasks import Task
from .fixtures.llm import LLM_PIPELINE
from .fixtures.task import CONTEXTS, TASKS


"""
What we want: 

- The task is initialized correctly using dataset
- The task contains a query using dataset
- The task contains a reference answer using dataset
"""

@pytest.mark.parametrize('task', TASKS)
def test_task_creation_with_dataset_context(task: Task):
    context = CONTEXTS[task]
    task(llm_pipeline=LLM_PIPELINE, context=context)
    assert task is not None

@pytest.mark.parametrize('task', TASKS)
def test_task_contains_query(task: Task):
    context = CONTEXTS[task]
    task = task(llm_pipeline=LLM_PIPELINE, context=context)
    assert task.query is not None

@pytest.mark.parametrize('task', TASKS)
def test_task_contains_reference(task: Task):
    context = CONTEXTS[task]
    task = task(llm_pipeline=LLM_PIPELINE, context=context)
    assert task.reference is not None

