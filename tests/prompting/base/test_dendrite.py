import pytest
from pydantic import ValidationError
from prompting import settings
settings.settings = settings.load_settings(mode="mock")

from prompting.base.dendrite import SynapseStreamResult, DendriteResponseEvent, StreamPromptingSynapse


def test_synapse_stream_result_creation():
    """Test the creation of SynapseStreamResult instances."""
    result = SynapseStreamResult(
        exception=None,
        uid=123,
        accumulated_chunks=["chunk1", "chunk2"],
        accumulated_chunks_timings=[0.1, 0.2],
        synapse=None,
    )

    assert result.uid == 123
    assert result.accumulated_chunks == ["chunk1", "chunk2"]
    assert result.accumulated_chunks_timings == [0.1, 0.2]
    assert result.synapse is None


def test_synapse_stream_result_completion_property():
    """Test the completion property of SynapseStreamResult."""
    # Case when synapse is None
    result = SynapseStreamResult(synapse=None)
    assert result.completion is None

    # Case when synapse is present
    synapse = StreamPromptingSynapse(
        task_name="test_task",
        roles=["assistant"],
        messages=["Hello"],
        completion="Test completion",
    )
    result.synapse = synapse
    assert result.completion == "Test completion"


def test_synapse_stream_result_model_dump():
    """Test the model_dump method of SynapseStreamResult."""
    synapse = StreamPromptingSynapse(
        task_name="test_task",
        roles=["assistant"],
        messages=["Hello"],
        completion="Test completion",
    )
    result = SynapseStreamResult(
        exception=Exception("Test exception"),
        uid=456,
        accumulated_chunks=["chunk1"],
        accumulated_chunks_timings=[0.1],
        synapse=synapse,
    )
    dumped = result.model_dump()
    assert dumped["uid"] == 456
    assert dumped["exception"].args[0] == "Test exception"
    assert dumped["accumulated_chunks"] == ["chunk1"]
    assert dumped["synapse"] == synapse.model_dump()


def test_stream_prompting_synapse_deserialize():
    """Test the deserialize method of StreamPromptingSynapse."""
    synapse = StreamPromptingSynapse(
        task_name="test_task",
        roles=["user"],
        messages=["Hello"],
        completion="Deserialized completion",
    )
    result = synapse.deserialize()
    assert result == "Deserialized completion"


def test_synapse_stream_result_exception_handling():
    """Test SynapseStreamResult when an exception is present."""
    exception = Exception("Test exception")
    result = SynapseStreamResult(
        exception=exception,
        uid=789,
        accumulated_chunks=None,
        accumulated_chunks_timings=None,
        synapse=None,
    )
    assert result.exception == exception
    assert result.uid == 789
    assert result.accumulated_chunks is None
    assert result.accumulated_chunks_timings is None
    assert result.completion is None

def test_dendrite_response_event_validation_error():
    """Test that DendriteResponseEvent raises a ValidationError when required fields are missing."""
    with pytest.raises(ValidationError):
        DendriteResponseEvent(
            uids=[1],
            timeout=5.0,
            stream_results=None,  # Missing stream_results
        )

def test_stream_prompting_synapse_required_fields():
    """Test that StreamPromptingSynapse raises an error when required fields are missing."""
    with pytest.raises(ValidationError):
        StreamPromptingSynapse(
            task_name="test_task",
            roles=["user"],
            messages=None,
        )

def test_stream_prompting_synapse_mutable_fields():
    """Test that immutable fields cannot be mutated."""
    synapse = StreamPromptingSynapse(
        task_name="test_task",
        roles=["user"],
        messages=["Hello"],
    )
    with pytest.raises(ValidationError):
        synapse.task_name = "new_task_name"

    with pytest.raises(ValidationError):
        synapse.roles = ["assistant"]

    with pytest.raises(ValidationError):
        synapse.messages = ["New message"]
