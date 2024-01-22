import pytest 
from prompting.agent import Persona
from prompting.agent import HumanAgent

from fixtures.llm import LLM_PIPELINE
from fixtures.tasks import CONTEXTS, TASKS

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
        - Persona contains a topic
        - Persona contains a subject
        - Persona contains a description
        - Persona contains a goal
        - Persona contains a query

    - Task is initialized correctly
        - Task contains a query
        - Task contains a reference
        - Task contains a context
        - Task contains a complete flag


rewards = 0.5 * (rouge(reference, completion)_bn + relevance(reference,completion)_bn) * (1 - rouge(challenge,completion)_bn)
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
def test_agent_contains_task(task: Task):
    context = CONTEXTS[task]
    task = task(llm_pipeline=LLM_PIPELINE, context=context)
    agent = HumanAgent(llm_pipeline=LLM_PIPELINE, task=task, begin_conversation=True)
    assert agent.task is not None

@pytest.mark.parametrize('task', TASKS)
def test_agent_can_make_queries(task: Task):
    context = CONTEXTS[task]
    task = task(llm_pipeline=LLM_PIPELINE, context=context)
    agent = HumanAgent(llm_pipeline=LLM_PIPELINE, task=task, begin_conversation=True)
    assert agent.query is not None

@pytest.mark.parametrize('task', TASKS)
def test_agent_can_make_challenges(task: Task):
    context = CONTEXTS[task]
    task = task(llm_pipeline=LLM_PIPELINE, context=context)
    agent = HumanAgent(llm_pipeline=LLM_PIPELINE, task=task)
    assert agent.challenge is not None
