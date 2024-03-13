import pytest
from prompting.llms import BaseLLM, BasePipeline, load_vllm_pipeline
from prompting.llms.utils import (
    contains_gpu_index_in_device,
    calculate_gpu_requirements,
)
from prompting.cleaners import CleanerPipeline
from prompting.mock import MockPipeline
from unittest import mock
from .fixtures.llm import llms, pipelines
from .fixtures.cleaner import DEFAULT_CLEANER_PIPELINE
import pytest
from unittest.mock import patch, MagicMock
from vllm import LLM


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
@mock.patch('torch.cuda.current_device', return_value=0)
@mock.patch('torch.cuda.synchronize')
@mock.patch("torch.cuda.mem_get_info")
def test_calculate_gpu_requirements(mock_mem_get_info, mock_synchronize, mock_current_device, device, max_allowed_memory_allocation_in_bytes, available_memory, expected_result):
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


# Test 1: Success on first attempt
@patch("prompting.llms.utils.calculate_gpu_requirements")
@patch("prompting.llms.vllm_llm.LLM")
def test_load_vllm_pipeline_success_first_try(
    mock_llm, mock_calculate_gpu_requirements
):
    # Mocking calculate_gpu_requirements to return a fixed value
    mock_calculate_gpu_requirements.return_value = 5e9  # Example value
    # Mocking LLM to return a mock LLM object without raising an exception
    mock_llm.return_value = MagicMock(spec=LLM)

    result = load_vllm_pipeline(model_id="test_name", device="cuda")
    assert isinstance(result, MagicMock)  # or any other assertion you find suitable
    mock_llm.assert_called_once()  # Ensures LLM was called exactly once


# # Test 2: Success on second attempt with larger memory allocation
@patch("prompting.llms.vllm_llm.clean_gpu_cache")
@patch("prompting.llms.utils.calculate_gpu_requirements")
@patch(
    "prompting.llms.vllm_llm.LLM",
    side_effect=[ValueError("First attempt failed"), MagicMock(spec=LLM)],
)
def test_load_vllm_pipeline_success_second_try(
    mock_llm, mock_calculate_gpu_requirements, mock_clean_gpu_cache
):
    mock_calculate_gpu_requirements.return_value = 5e9  # Example value for both calls

    result = load_vllm_pipeline(model_id="test", device="cuda")
    assert isinstance(result, MagicMock)
    assert mock_llm.call_count == 2  # LLM is called twice
    mock_clean_gpu_cache.assert_called_once()  # Ensures clean_gpu_cache was called


# # Test 3: Exception on second attempt
@patch("prompting.llms.vllm_llm.clean_gpu_cache")
@patch("prompting.llms.utils.calculate_gpu_requirements")
@patch(
    "prompting.llms.vllm_llm.LLM",
    side_effect=[
        ValueError("First attempt failed"),
        Exception("Second attempt failed"),
    ],
)
def test_load_vllm_pipeline_exception_second_try(
    mock_llm, mock_calculate_gpu_requirements, mock_clean_gpu_cache
):
    mock_calculate_gpu_requirements.return_value = (
        5e9  # Example value for both attempts
    )

    with pytest.raises(Exception, match="Second attempt failed"):
        load_vllm_pipeline(model_id="HuggingFaceH4/zephyr-7b-beta", device="gpu0")
    assert mock_llm.call_count == 2  # LLM is called twice
    mock_clean_gpu_cache.assert_called_once()  # Ensures clean_gpu_cache was called
