import pytest
from prompting.tasks import Task, QuestionAnsweringTask, SummarizationTask, DebuggingTask, MathTask, DateQuestionAnsweringTask
from prompting.mock import MockPipeline

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


LLM_PIPELINE = MockPipeline("mock")
CONTEXT = {"text": "This is a context.", "title": "this is a title"}

TASKS = [
        QuestionAnsweringTask,
        SummarizationTask,
        DebuggingTask,
        MathTask,
        DateQuestionAnsweringTask,
    ]
CONTEXTS = {
    QuestionAnsweringTask: {"text": "This is a context.", "title": "this is a title", "categories": ['some','categories']},
    SummarizationTask: {"text": "This is a context.", "title": "this is a title", "categories": ['some','categories']},
    DebuggingTask: {"code": "This is code","repo_name":'prompting',"path":'this/is/a/path', "language":'python'},
    MathTask: {"problem": "This is a problem","solution":'3.1415'},
    DateQuestionAnsweringTask: {"section": "Events", "event":"1953 - Battle of Hastings in UK", 'date':"1 January"},
}

# TODO: Math task only works when solution is floatable
# TODO: DateQA only accepts section in {Births, Deaths, Events}
# TODO: DateQA expect wiki entry for event 

@pytest.mark.parametrize('task', TASKS)
def test_create_task(task: Task):
    context = CONTEXTS[task]
    task(llm_pipeline=LLM_PIPELINE, context=context)

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

# @pytest.mark.parametrize('task', TASKS)
# def test_task_contains_reward_definition(task: Task):
#     context = CONTEXTS[task]
#     task = task(llm_pipeline=LLM_PIPELINE, context=context)
#     assert task.reward_definition is not None    

# @pytest.mark.parametrize('task', TASKS)
# def test_task_contains_goal(task: Task):
#     context = CONTEXTS[task]
#     task = task(llm_pipeline=LLM_PIPELINE, context=context)
#     assert task.goal is not None

# @pytest.mark.parametrize('task', TASKS)
# def test_task_contains_desc(task: Task):
#     context = CONTEXTS[task]
#     task = task(llm_pipeline=LLM_PIPELINE, context=context)
#     assert task.desc is not None

# @pytest.mark.parametrize('task', TASKS)
# def test_task_contains_query_time(task: Task):
#     context = CONTEXTS[task]
#     task = task(llm_pipeline=LLM_PIPELINE, context=context)
#     assert task.reference_time>=0

# @pytest.mark.parametrize('task', TASKS)
# def test_task_contains_reference_time(task: Task):
#     context = CONTEXTS[task]
#     task = task(llm_pipeline=LLM_PIPELINE, context=context)
#     assert task.query_time>=0
