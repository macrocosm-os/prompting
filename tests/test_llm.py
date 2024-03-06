import pytest
import torch
from prompting.llms import BaseLLM, BasePipeline
from prompting.llms.utils import (
    contains_gpu_index_in_device,
    calculate_gpu_requirements,
)
from prompting.cleaners import CleanerPipeline
from prompting.mock import MockPipeline
from unittest import mock
from .fixtures.llm import llms, pipelines
from .fixtures.cleaner import DEFAULT_CLEANER_PIPELINE


@pytest.mark.parametrize(
    "input, expected_result, cleaner",
    [
        (
            '"I am a quote. User: I know you are. I am asking a question. What is th"',
            '"I am a quote. User: I know you are. I am asking a question. What is th"',
            None,
        ),
        (
            '"I am a quote. User: I know you are. I am asking a question. What is th"',
            "I am a quote. I know you are. I am asking a question.",
            DEFAULT_CLEANER_PIPELINE,
        ),
    ],
)
@pytest.mark.parametrize("llm", llms())
def test_llm_clean_response(
    input: str, expected_result: str, cleaner: CleanerPipeline, llm: BaseLLM
):
    result = llm.clean_response(cleaner=cleaner, response=input)
    assert result == expected_result


@pytest.mark.parametrize("pipeline", pipelines())
def test_load_pipeline_mock(pipeline: BasePipeline):
    # Note that the model_id will be used internally as static response for the mock pipeline
    model_id = "gpt2"
    pipeline_instance = pipeline(model_id=model_id, device="cpu", mock=True)
    pipeline_message = pipeline_instance("")

    mock_message = MockPipeline(model_id).forward(messages=[])
    assert mock_message == pipeline_message


@pytest.mark.parametrize("llm", llms())
def test_llm_query(llm: BaseLLM):
    message = "test"
    llm.query(message)

    # Assert that stateful operation where 3 messages are saved:
    # the system prompt (on llm init), the user message and the assistant reply
    assert len(llm.messages) == 3
    assert len(llm.times) == 3

    assert llm.messages[0]["role"] == "system"

    assert llm.messages[1]["role"] == "user"
    assert llm.messages[1]["content"] == message

    assert llm.messages[2]["role"] == "assistant"


@pytest.mark.parametrize("llm", llms())
def test_llm_forward(llm: BaseLLM):
    llm.forward(llm.messages)

    # Assert stateless operation of the model with only history of system prompt
    assert len(llm.messages) == 1
    assert len(llm.times) == 1
    assert llm.messages[0]["role"] == "system"


@pytest.mark.parametrize(
    "device, expected_result", [("cpu", False), ("cuda", False), ("cuda:0", True)]
)
def test_contains_gpu_index_in_device(device: str, expected_result: bool):
    result = contains_gpu_index_in_device(device)
    assert result == expected_result


@pytest.mark.parametrize(
    "device, max_allowed_memory_allocation_in_bytes, available_memory, expected_result",
    [
        ("cuda", 20e9, 20e9, 1),
        ("cuda", 20e9, 40e9, 0.5),
        ("cuda:0", 40e9, 160e9, 0.25),
    ],
)
@mock.patch("torch.cuda.mem_get_info")
def test_calculate_gpu_requirements(
    mock_mem_get_info,
    device: str,
    max_allowed_memory_allocation_in_bytes: int,
    available_memory: float,
    expected_result: float,
):
    mock_mem_get_info.return_value = (available_memory, available_memory)
    result = calculate_gpu_requirements(device, max_allowed_memory_allocation_in_bytes)
    assert result == expected_result


@pytest.mark.parametrize(
    "available_memory, max_allowed_memory_allocation_in_bytes",
    [(10e9, 20e9), (20e9, 40e9)],
)
@mock.patch("torch.cuda.mem_get_info")
def test_calulate_gpu_requirements_raises_cuda_error(
    mock_mem_get_info,
    available_memory: float,
    max_allowed_memory_allocation_in_bytes: float,
):
    mock_mem_get_info.return_value = (available_memory, available_memory)

    with pytest.raises(Exception):
        calculate_gpu_requirements("cuda", max_allowed_memory_allocation_in_bytes)
