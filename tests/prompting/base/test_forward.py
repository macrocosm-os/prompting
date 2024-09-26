import pytest
from unittest.mock import Mock, AsyncMock

from prompting import settings
settings.settings = settings.load_settings(mode="mock")

from prompting.base.forward import execute_dendrite_call, generate_reference, handle_response, process_stream
from prompting.base.protocol import StreamPromptingSynapse
from prompting.llms.base_llm import BasePipeline
from prompting.tasks.base_task import BaseTextTask


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "return_value, expected_result",
    [
        ("response", "response"),
        (42, 42),
        ({"key": "value"}, {"key": "value"}),
    ]
)
async def test_execute_dendrite_call(return_value, expected_result):
    """Test the execute_dendrite_call function with different return values."""
    mock_response = AsyncMock(return_value=return_value)
    result = await execute_dendrite_call(mock_response())
    assert result == expected_result


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "side_effect, expected_exception_type, expected_message",
    [
        (Exception("Test exception"), Exception, "Test exception"),
        (ValueError("Invalid value"), ValueError, "Invalid value"),
    ]
)
async def test_execute_dendrite_call_exception(side_effect, expected_exception_type, expected_message):
    """Test the execute_dendrite_call function when an exception occurs."""
    mock_response = AsyncMock(side_effect=side_effect)
    with pytest.raises(expected_exception_type) as exc_info:
        await execute_dendrite_call(mock_response())
    assert str(exc_info.value) == expected_message


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "uid, iterator_content, expected_exception, expected_accumulated_chunks, expected_completion",
    [
        # Successful stream processing.
        (
            1,
            [
                "chunk1",
                "chunk2",
                StreamPromptingSynapse(
                    task_name="task",
                    roles=["assistant"],
                    messages=["Hello"],
                    completion="Completion",
                    required_hash_fields=["hash"]
                ),
            ],
            None,
            ["chunk1", "chunk2"],
            "Completion",
        ),
        # Stream processing with exception.
        (
            2,
            [
                "chunk1",
                Exception("Test exception"),
            ],
            Exception("Test exception"),
            ["chunk1"],
            None,
        ),
    ]
)
async def test_process_stream(uid, iterator_content, expected_exception, expected_accumulated_chunks, expected_completion):
    """Test the process_stream function with various scenarios."""
    async def mock_async_iterator():
        for item in iterator_content:
            if isinstance(item, Exception):
                raise item
            else:
                yield item

    result = await process_stream(uid, mock_async_iterator())

    assert result.uid == uid
    if expected_exception is None:
        assert result.exception is None
    else:
        assert result.exception is not None
        assert isinstance(result.exception, type(expected_exception))
        assert str(result.exception) == str(expected_exception)

    assert result.accumulated_chunks == expected_accumulated_chunks
    if expected_completion is not None:
        assert result.completion == expected_completion
    else:
        assert result.completion is None


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "stream_results_dict, expected_results",
    [
        # Both streams succeed.
        (
            {
                1: [
                    "chunk1",
                    "chunk2",
                    StreamPromptingSynapse(
                        task_name="task",
                        roles=["assistant"],
                        messages=["Hello"],
                        completion="Completion 1",
                        required_hash_fields=["hash"]
                    ),
                ],
                2: [
                    "chunkA",
                    "chunkB",
                    StreamPromptingSynapse(
                        task_name="task",
                        roles=["assistant"],
                        messages=["Hi"],
                        completion="Completion 2",
                        required_hash_fields=["hash"]
                    ),
                ],
            },
            [
                {
                    "uid": 1,
                    "exception": None,
                    "accumulated_chunks": ["chunk1", "chunk2"],
                    "completion": "Completion 1",
                },
                {
                    "uid": 2,
                    "exception": None,
                    "accumulated_chunks": ["chunkA", "chunkB"],
                    "completion": "Completion 2",
                },
            ],
        ),
        # One stream raises an exception.
        (
            {
                1: [
                    "chunk1",
                    Exception("Stream 1 exception"),
                ],
                2: [
                    "chunkA",
                    StreamPromptingSynapse(
                        task_name="task",
                        roles=["assistant"],
                        messages=["Hi"],
                        completion="Completion 2",
                        required_hash_fields=["hash"]
                    ),
                ],
            },
            [
                {
                    "uid": 1,
                    "exception": Exception("Stream 1 exception"),
                    "accumulated_chunks": ["chunk1"],
                    "completion": None,
                },
                {
                    "uid": 2,
                    "exception": None,
                    "accumulated_chunks": ["chunkA"],
                    "completion": "Completion 2",
                },
            ],
        ),
    ]
)
async def test_handle_response(stream_results_dict, expected_results):
    """Test the handle_response function with various scenarios."""
    def make_mock_async_iterator(content):
        async def iterator():
            for item in content:
                if isinstance(item, Exception):
                    raise item
                else:
                    yield item
        return iterator()

    # Prepare the stream_results_dict with async iterators.
    stream_results = {uid: make_mock_async_iterator(content) for uid, content in stream_results_dict.items()}

    results = await handle_response(stream_results)

    assert len(results) == len(expected_results)
    for result, expected in zip(results, expected_results):
        assert result.uid == expected["uid"]
        if expected["exception"] is None:
            assert result.exception is None
        else:
            assert result.exception is not None
            assert isinstance(result.exception, type(expected["exception"]))
            assert str(result.exception) == str(expected["exception"])

        assert result.accumulated_chunks == expected["accumulated_chunks"]
        if expected["completion"] is not None:
            assert result.synapse is not None
            assert result.synapse.completion == expected["completion"]
        else:
            assert result.synapse is None or result.synapse.completion is None


@pytest.mark.asyncio
async def test_generate_reference():
    """Test the generate_reference function."""
    class MockTask(BaseTextTask):
        def generate_reference(self, pipeline):
            return "Reference text"

    mock_task = MockTask()
    mock_pipeline = Mock(spec=BasePipeline)

    result = await generate_reference(mock_task, mock_pipeline)
    assert result == "Reference text"
