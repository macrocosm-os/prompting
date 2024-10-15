import pytest
from pydantic import ValidationError

from prompting import settings

settings.settings = settings.Settings.load(mode="mock")

from prompting.base.dendrite import DendriteResponseEvent, StreamPromptingSynapse, SynapseStreamResult


@pytest.mark.parametrize(
    "exception, uid, accumulated_chunks, accumulated_chunks_timings, synapse",
    [
        (None, 123, ["chunk1", "chunk2"], [0.1, 0.2], None),
        (Exception("Test exception"), 456, ["chunk3"], [0.3], None),
        (None, 789, [], [], StreamPromptingSynapse(
            task_name="test_task",
            roles=["assistant"],
            messages=["Hello"],
            completion="Test completion",
        )),
    ]
)
def test_synapse_stream_result_creation(exception, uid, accumulated_chunks, accumulated_chunks_timings, synapse):
    """Test the creation of SynapseStreamResult instances with various inputs."""
    result = SynapseStreamResult(
        exception=exception,
        uid=uid,
        accumulated_chunks=accumulated_chunks,
        accumulated_chunks_timings=accumulated_chunks_timings,
        synapse=synapse,
    )

    assert result.uid == uid
    assert result.accumulated_chunks == accumulated_chunks
    assert result.accumulated_chunks_timings == accumulated_chunks_timings
    assert result.synapse == synapse


@pytest.mark.parametrize(
    "synapse, expected_completion",
    [
        (None, None),
        (
            StreamPromptingSynapse(
                task_name="test_task",
                roles=["assistant"],
                messages=["Hello"],
                completion="Test completion",
            ),
            "Test completion",
        ),
    ]
)
def test_synapse_stream_result_completion_property(synapse, expected_completion):
    """Test the completion property of SynapseStreamResult with various synapse inputs."""
    result = SynapseStreamResult(synapse=synapse)
    assert result.completion == expected_completion


@pytest.mark.parametrize(
    "exception, uid, accumulated_chunks, accumulated_chunks_timings, synapse",
    [
        (
            Exception("Test exception"),
            456,
            ["chunk1"],
            [0.1],
            StreamPromptingSynapse(
                task_name="test_task",
                roles=["assistant"],
                messages=["Hello"],
                completion="Test completion",
            ),
        ),
        (
            None,
            789,
            [],
            [],
            None,
        ),
    ]
)
def test_synapse_stream_result_model_dump(exception, uid, accumulated_chunks, accumulated_chunks_timings, synapse):
    """Test the model_dump method of SynapseStreamResult with various inputs."""
    result = SynapseStreamResult(
        exception=exception,
        uid=uid,
        accumulated_chunks=accumulated_chunks,
        accumulated_chunks_timings=accumulated_chunks_timings,
        synapse=synapse,
    )
    dumped = result.model_dump()

    assert dumped["uid"] == uid
    if exception:
        assert dumped["exception"].args[0] == exception.args[0]
    else:
        assert dumped["exception"] is None
    assert dumped["accumulated_chunks"] == accumulated_chunks
    if synapse is not None:
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


@pytest.mark.parametrize(
    "exception, uid, accumulated_chunks, accumulated_chunks_timings, synapse",
    [
        (
            Exception("Test exception"),
            789,
            None,
            None,
            None,
        ),
        (
            ValueError("Value error occurred"),
            101,
            ["chunk1"],
            [0.1],
            None,
        ),
    ]
)
def test_synapse_stream_result_exception_handling(exception, uid, accumulated_chunks, accumulated_chunks_timings, synapse):
    """Test SynapseStreamResult when an exception is present."""
    result = SynapseStreamResult(
        exception=exception,
        uid=uid,
        accumulated_chunks=accumulated_chunks,
        accumulated_chunks_timings=accumulated_chunks_timings,
        synapse=synapse,
    )
    assert result.exception == exception
    assert result.uid == uid
    assert result.accumulated_chunks == accumulated_chunks
    assert result.accumulated_chunks_timings == accumulated_chunks_timings
    assert result.completion is None


@pytest.mark.parametrize(
    "uids, timeout, stream_results",
    [
        # Missing uids.
        (None, 5.0, None),
        # Missing stream_results.
        ([1], 5.0, None),
         # Missing all.
        (None, None, None),
    ]
)
def test_dendrite_response_event_validation_error(uids, timeout, stream_results):
    """Test that DendriteResponseEvent raises a ValidationError when required fields are missing."""
    with pytest.raises(ValidationError):
        DendriteResponseEvent(
            uids=uids,
            timeout=timeout,
            stream_results=stream_results,
        )


@pytest.mark.parametrize(
    "task_name, roles, messages",
    [
        # Missing task_name.
        (None, ["user"], ["Hello"]),
        # Missing roles.
        ("test_task", None, ["Hello"]),
        # Missing messages.
        ("test_task", ["user"], None),
         # Missing all.
        (None, None, None),
    ]
)
def test_stream_prompting_synapse_required_fields(task_name, roles, messages):
    """Test that StreamPromptingSynapse raises an error when required fields are missing."""
    with pytest.raises(ValidationError):
        StreamPromptingSynapse(
            task_name=task_name,
            roles=roles,
            messages=messages,
        )


@pytest.mark.parametrize(
    "field_name, new_value",
    [
        ("task_name", "new_task_name"),
        ("roles", ["assistant"]),
        ("messages", ["New message"]),
    ]
)
def test_stream_prompting_synapse_mutable_fields(field_name, new_value):
    """Test that immutable fields cannot be mutated."""
    synapse = StreamPromptingSynapse(
        task_name="test_task",
        roles=["user"],
        messages=["Hello"],
    )
    with pytest.raises((ValidationError, AttributeError, TypeError)):
        setattr(synapse, field_name, new_value)