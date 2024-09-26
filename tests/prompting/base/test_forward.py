import pytest
from unittest.mock import Mock, AsyncMock

from prompting import settings
settings.settings = settings.load_settings(mode="mock")

from prompting.base.forward import execute_dendrite_call, generate_reference, handle_response, process_stream
from prompting.base.protocol import StreamPromptingSynapse
from prompting.llms.base_llm import BasePipeline
from prompting.tasks.base_task import BaseTextTask


@pytest.mark.asyncio
async def test_execute_dendrite_call():
    """Test the execute_dendrite_call function."""
    mock_response = AsyncMock(return_value="response")
    result = await execute_dendrite_call(mock_response())
    assert result == "response"


@pytest.mark.asyncio
async def test_process_stream_success():
    """Test the process_stream function with successful stream."""
    uid = 1

    async def mock_async_iterator():
        yield "chunk1"
        yield "chunk2"
        yield StreamPromptingSynapse(task_name="task", roles=["assistant"], messages=["Hello"], completion="Completion",
                                     required_hash_fields=["hash"])

    result = await process_stream(uid, mock_async_iterator())

    assert result.uid == uid
    assert result.exception is None
    assert result.accumulated_chunks == ["chunk1", "chunk2"]
    assert isinstance(result.synapse, StreamPromptingSynapse)
    assert result.synapse.completion == "Completion"


@pytest.mark.asyncio
async def test_process_stream_exception():
    """Test the process_stream function when an exception occurs."""
    uid = 2

    async def mock_async_iterator():
        yield "chunk1"
        raise Exception("Test exception")

    result = await process_stream(uid, mock_async_iterator())

    assert result.uid == uid
    assert result.exception is not None
    assert isinstance(result.exception, Exception)
    assert str(result.exception) == "Test exception"
    assert result.accumulated_chunks == ["chunk1"]


@pytest.mark.asyncio
async def test_handle_response():
    """Test the handle_response function."""
    async def mock_async_iterator1():
        yield "chunk1"
        yield "chunk2"
        yield StreamPromptingSynapse(task_name="task", roles=["assistant"], messages=["Hello"], completion="Completion 1",
                                     required_hash_fields=["hash"])

    async def mock_async_iterator2():
        yield "chunkA"
        yield "chunkB"
        yield StreamPromptingSynapse(task_name="task", roles=["assistant"], messages=["Hi"], completion="Completion 2",
                                     required_hash_fields=["hash"])

    stream_results_dict = {
        1: mock_async_iterator1(),
        2: mock_async_iterator2(),
    }

    results = await handle_response(stream_results_dict)

    assert len(results) == 2

    result1 = results[0]
    assert result1.uid == 1
    assert result1.exception is None
    assert result1.accumulated_chunks == ["chunk1", "chunk2"]
    assert result1.synapse.completion == "Completion 1"

    result2 = results[1]
    assert result2.uid == 2
    assert result2.exception is None
    assert result2.accumulated_chunks == ["chunkA", "chunkB"]
    assert result2.synapse.completion == "Completion 2"


@pytest.mark.asyncio
async def test_handle_response_with_exception():
    """Test handle_response when one of the streams raises an exception."""
    async def mock_async_iterator1():
        yield "chunk1"
        raise Exception("Stream 1 exception")

    async def mock_async_iterator2():
        yield "chunkA"
        yield StreamPromptingSynapse(task_name="task", roles=["assistant"], messages=["Hi"], completion="Completion 2",
                                     required_hash_fields=["hash"])

    stream_results_dict = {
        1: mock_async_iterator1(),
        2: mock_async_iterator2(),
    }

    results = await handle_response(stream_results_dict)

    assert len(results) == 2

    result1 = results[0]
    assert result1.uid == 1
    assert result1.exception is not None
    assert str(result1.exception) == "Stream 1 exception"
    assert result1.accumulated_chunks == ["chunk1"]

    result2 = results[1]
    assert result2.uid == 2
    assert result2.exception is None
    assert result2.accumulated_chunks == ["chunkA"]
    assert result2.synapse.completion == "Completion 2"

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
