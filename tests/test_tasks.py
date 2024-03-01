import pytest
import inspect
from inspect import signature
from prompting.tasks import Task
from prompting.rewards import REWARD_MODELS
from .fixtures.task import CONTEXTS, TASKS, TASK_FIELDS
from .fixtures.llm import LLM_PIPELINE

# TODO: Check if format_challenge is defined
# TODO: Ensure that when static_reference is True, reference_time is not defined. Same for query_time and static_query
# TODO: Ensure that when generate_reference=True and static_reference is True,there is still a reference
# TODO: Ensure that when generate_reference=False and static_reference is True,there is still a reference
# TODO: Ensure that when generate_reference=False and static_reference is False,there is NOT a reference


@pytest.mark.parametrize('task', TASKS)
def test_create_task(task: Task):
    task(llm_pipeline=LLM_PIPELINE, context=CONTEXTS[task])


@pytest.mark.parametrize('task', TASKS)
@pytest.mark.parametrize('field', TASK_FIELDS.keys())
def test_task_contains_expected_field(task: Task, field: str):
    task = task(llm_pipeline=LLM_PIPELINE, context=CONTEXTS[task])
    assert hasattr(task, field)


@pytest.mark.parametrize('task', TASKS)
@pytest.mark.parametrize('field, expected_type', list(TASK_FIELDS.items()))
def test_task_field_has_expected_type(task: Task, field: str, expected_type: type):
    task = task(llm_pipeline=LLM_PIPELINE, context=CONTEXTS[task])
    assert isinstance(getattr(task, field), expected_type)


@pytest.mark.parametrize('task', TASKS)
@pytest.mark.parametrize('field', TASK_FIELDS.keys())
def test_task_field_is_not_null(task: Task, field: str):
    task = task(llm_pipeline=LLM_PIPELINE, context=CONTEXTS[task])
    assert getattr(task, field) is not None


@pytest.mark.parametrize('task', TASKS)
def test_task_complete_is_false_on_init(task: Task):

    task = task(llm_pipeline=LLM_PIPELINE, context=CONTEXTS[task])
    assert task.complete == False


@pytest.mark.parametrize('task', TASKS)
def test_task_contains_no_reference_if_not_static(task: Task):
    task(llm_pipeline=LLM_PIPELINE, context=CONTEXTS[task], create_reference=False)
    assert task.static_reference or not task.reference


@pytest.mark.parametrize('task', TASKS)
def test_task_contains_query_time(task: Task):

    task = task(llm_pipeline=LLM_PIPELINE, context=CONTEXTS[task])
    assert task.static_reference or task.reference_time>=0


@pytest.mark.parametrize('task', TASKS)
def test_task_contains_reference_time(task: Task):

    task = task(llm_pipeline=LLM_PIPELINE, context=CONTEXTS[task])
    assert task.static_query or task.query_time>=0


@pytest.mark.parametrize('task', TASKS)
@pytest.mark.parametrize('full', (True, False))
def test_task_state_dict(task: Task, full: bool):

    task = task(llm_pipeline=LLM_PIPELINE, context=CONTEXTS[task])
    assert type(task.__state_dict__(full)) == dict


@pytest.mark.parametrize('task', TASKS)
@pytest.mark.parametrize('definition, expected_weight', [('reward_definition',1), ('penalty_definition',None)])
def test_task_contains_required_definition(task: Task, definition: str, expected_weight: float):

    task = task(llm_pipeline=LLM_PIPELINE, context=CONTEXTS[task])
    model_infos = getattr(task, definition)
    total_weight = 0
    for model_info in model_infos:

        assert isinstance(model_info, dict)

        name = model_info.get("name")
        assert name is not None
        assert name in REWARD_MODELS

        params = {k: v for k, v in model_info.items() if k not in ["name", "weight"]}
        cls_params = signature(REWARD_MODELS['rouge']).parameters
        # check that all the parameters are in the class (no extra parameters are allowed)
        for k, v in params.items():
            assert k in cls_params
            # check that the type of the parameter is correct or not annotated
            assert cls_params[k].annotation == inspect._empty or isinstance(v, cls_params[k].annotation)

        # check that all class parameters without default values are in the model_info
        for k, v in cls_params.items():
            # ignore self, device, args, kwargs
            if v.default == inspect._empty and v.name not in ("self", "device", "args", "kwargs"):
                assert k in params

        weight = model_info.get("weight")
        assert weight is not None
        assert isinstance(weight, (float, int))
        assert 0 <= weight <= 1

        total_weight += weight

    assert not model_infos or expected_weight is None or total_weight == expected_weight