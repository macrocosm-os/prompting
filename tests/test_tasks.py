import pytest

from prompting.tasks import Task
from .fixtures.task import CONTEXTS, TASKS
from .fixtures.llm import LLM_PIPELINE

"""
What we want to test for each task:
- The task is initialized correctly
- The task contains a query
- The task contains a reference answer
- Task contains a query_time
- Task contains a reference_time
- The task formats correctly
- All task fields are present as expected
- Tasks have reward definitions
"""


# TODO: Math task only works when solution is floatable
# TODO: DateQA only accepts section in {Births, Deaths, Events}
# TODO: DateQA expect wiki entry for event

@pytest.mark.parametrize('task', TASKS)
def test_create_task(task: Task):

    task(llm_pipeline=LLM_PIPELINE, context=CONTEXTS[task].copy())

@pytest.mark.parametrize('task', TASKS)
def test_task_contains_query(task: Task):

    task = task(llm_pipeline=LLM_PIPELINE, context=CONTEXTS[task].copy())
    assert task.query is not None

@pytest.mark.parametrize('task', TASKS)
def test_task_contains_reference(task: Task):

    task = task(llm_pipeline=LLM_PIPELINE, context=CONTEXTS[task].copy())
    assert task.reference is not None

@pytest.mark.parametrize('task', TASKS)
def test_task_contains_reward_definition(task: Task):

    task = task(llm_pipeline=LLM_PIPELINE, context=CONTEXTS[task].copy())
    assert type(task.reward_definition) == list


@pytest.mark.parametrize('task', TASKS)
def test_task_contains_goal(task: Task):

    task = task(llm_pipeline=LLM_PIPELINE, context=CONTEXTS[task].copy())
    assert task.goal is not None

@pytest.mark.parametrize('task', TASKS)
def test_task_contains_desc(task: Task):

    task = task(llm_pipeline=LLM_PIPELINE, context=CONTEXTS[task].copy())
    assert task.desc is not None

@pytest.mark.parametrize('task', TASKS)
def test_task_complete_is_false_on_init(task: Task):

    task = task(llm_pipeline=LLM_PIPELINE, context=CONTEXTS[task].copy())
    assert task.complete == False

@pytest.mark.parametrize('task', TASKS)
def test_task_contains_tags(task: Task):

    task = task(llm_pipeline=LLM_PIPELINE, context=CONTEXTS[task].copy())
    assert type(task.tags) == list

@pytest.mark.parametrize('task', TASKS)
def test_task_contains_context(task: Task):
    context = CONTEXTS[task].copy()
    task = task(llm_pipeline=LLM_PIPELINE, context=CONTEXTS[task].copy())
    assert context == task.context

@pytest.mark.parametrize('task', TASKS)
def test_task_contains_query_time(task: Task):

    task = task(llm_pipeline=LLM_PIPELINE, context=CONTEXTS[task].copy())
    assert task.static_reference or task.reference_time>=0

@pytest.mark.parametrize('task', TASKS)
def test_task_contains_reference_time(task: Task):

    task = task(llm_pipeline=LLM_PIPELINE, context=CONTEXTS[task].copy())
    assert task.static_query or task.query_time>=0
