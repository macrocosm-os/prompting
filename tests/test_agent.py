import pytest
from prompting.tasks import Task
from prompting.agent import HumanAgent, create_persona

from .fixtures.llm import LLM_PIPELINE
from .fixtures.task import CONTEXTS, TASKS

"""
Things to test:

- Agent is initialized correctly
    - Agent contains a persona
    - Agent contains a task
    - Agent can make queries
    - Agent can make responses

- Persona is initialized correctly
    - Persona contains a mood
    - Persona contains a tone

- Task is initialized correctly
    - Task contains a query
    - Task contains a reference
    - Task contains a context
    - Task contains a complete flag
        llm_pipeline: Pipeline,
        : str = None,
        persona: Persona = None,
        begin_conversation=True,
"""


@pytest.mark.parametrize('task', TASKS)
def test_agent_creation_with_dataset_context(task: Task):
    context = CONTEXTS[task]
    task = task(llm_pipeline=LLM_PIPELINE, context=context)
    agent = HumanAgent(llm_pipeline=LLM_PIPELINE, task=task, begin_conversation=True)
    assert agent is not None

@pytest.mark.parametrize('task', TASKS)
def test_agent_contains_persona(task: Task):
    context = CONTEXTS[task]
    task = task(llm_pipeline=LLM_PIPELINE, context=context)
    agent = HumanAgent(llm_pipeline=LLM_PIPELINE, task=task, begin_conversation=True)
    assert agent.persona is not None

@pytest.mark.parametrize('task', TASKS)
def test_user_can_set_agent_persona(task: Task):
    context = CONTEXTS[task]
    persona = create_persona()
    task = task(llm_pipeline=LLM_PIPELINE, context=context)
    agent = HumanAgent(llm_pipeline=LLM_PIPELINE, task=task, begin_conversation=True, persona=persona)
    assert agent.persona == persona

@pytest.mark.parametrize('task', TASKS)
def test_agent_contains_task(task: Task):
    context = CONTEXTS[task]
    task = task(llm_pipeline=LLM_PIPELINE, context=context)
    agent = HumanAgent(llm_pipeline=LLM_PIPELINE, task=task, begin_conversation=True)
    assert agent.task is not None

@pytest.mark.parametrize('task', TASKS)
def test_agent_has_system_prompt(task: Task):
    context = CONTEXTS[task]
    task = task(llm_pipeline=LLM_PIPELINE, context=context)
    agent = HumanAgent(llm_pipeline=LLM_PIPELINE, task=task, begin_conversation=True)
    assert agent.system_prompt is not None

@pytest.mark.parametrize('task', TASKS)
def test_user_can_set_agent_system_prompt_template(task: Task):
    context = CONTEXTS[task]
    system_template = "Today I am in a {mood} mood because i wanted {desc} related to {topic} ({subtopic}) in a {tone} tone. My intention is {goal}, but my problem is {query}"

    task = task(llm_pipeline=LLM_PIPELINE, context=context)
    agent = HumanAgent(llm_pipeline=LLM_PIPELINE, task=task, begin_conversation=True, system_template=system_template)
    assert agent.system_prompt_template


@pytest.mark.parametrize('task', TASKS)
@pytest.mark.parametrize('begin_conversation', [True, False])
def test_agent_can_make_challenges(task: Task, begin_conversation: bool):
    context = CONTEXTS[task]
    task = task(llm_pipeline=LLM_PIPELINE, context=context)
    agent = HumanAgent(llm_pipeline=LLM_PIPELINE, task=task, begin_conversation=begin_conversation)
    if begin_conversation:
        assert agent.challenge is not None
    else:
        assert getattr(agent, 'challenge', None) is None

@pytest.mark.parametrize('task', TASKS)
def test_agent_progress_is_zero_on_init(task: Task):
    context = CONTEXTS[task]
    task = task(llm_pipeline=LLM_PIPELINE, context=context)
    agent = HumanAgent(llm_pipeline=LLM_PIPELINE, task=task, begin_conversation=True)
    assert agent.progress == 0

@pytest.mark.parametrize('task', TASKS)
def test_agent_progress_is_one_when_task_is_complete(task: Task):
    context = CONTEXTS[task]
    task = task(llm_pipeline=LLM_PIPELINE, context=context)
    task.complete = True
    agent = HumanAgent(llm_pipeline=LLM_PIPELINE, task=task, begin_conversation=True)
    assert agent.progress == 1

@pytest.mark.parametrize('task', TASKS)
def test_agent_finished_is_true_when_task_is_complete(task: Task):
    context = CONTEXTS[task]
    task = task(llm_pipeline=LLM_PIPELINE, context=context)
    task.complete = True
    agent = HumanAgent(llm_pipeline=LLM_PIPELINE, task=task, begin_conversation=True)
    assert agent.finished == True

@pytest.mark.parametrize('task', TASKS)
def test_agent_finished_is_false_when_task_is_not_complete(task: Task):
    context = CONTEXTS[task]
    task = task(llm_pipeline=LLM_PIPELINE, context=context)
    task.complete = False
    agent = HumanAgent(llm_pipeline=LLM_PIPELINE, task=task, begin_conversation=True)
    assert agent.finished == False
