import pytest
from pydantic import ValidationError

from validator_api.serializers import ChatCompletionMessage, Choice, CompletionsRequest, CompletionsResponse


def test_standard_inference_request_validation():
    """Test that the standard inference request validation works correctly."""
    # Valid request
    valid_request = {
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "What's the biggest crab in the world?"},
        ],
        "model": "hugging-quants/Meta-Llama-3.1-70B-Instruct-AWQ-INT4",
        "task": "InferenceTask",
        "test_time_inference": False,
        "mixture": False,
        "sampling_parameters": {"temperature": 0.7, "top_p": 0.95, "do_sample": True, "max_new_tokens": 256},
        "stream": False,
    }

    request = CompletionsRequest(**valid_request)
    assert request.messages == valid_request["messages"]
    assert request.model == valid_request["model"]
    assert request.sampling_parameters == valid_request["sampling_parameters"]

    # Test missing required messages field
    with pytest.raises(ValidationError) as exc_info:
        CompletionsRequest()
    assert "messages" in str(exc_info.value)

    # Test invalid message format
    with pytest.raises(ValueError) as exc_info:
        CompletionsRequest(messages=[{"invalid_key": "content"}])
    assert "must contain 'role' and 'content' keys" in str(exc_info.value)


def test_standard_inference_response_validation():
    """Test that the standard inference response validation works correctly."""
    valid_response = {
        "id": "1dd0f0dc-bbc9-4be3-b143-c5340672e77e",
        "choices": [
            {
                "finish_reason": "stop",
                "index": 0,
                "message": {
                    "content": "The Japanese spider crab (Macrocheira kaempferi) is considered the largest crab species in the world.",
                    "role": "assistant",
                },
            }
        ],
        "created": 1741960995,
        "model": "hugging-quants/Meta-Llama-3.1-70B-Instruct-AWQ-INT4",
        "object": "chat.completion",
    }

    response = CompletionsResponse(**valid_response)
    assert response.id == valid_response["id"]
    assert len(response.choices) == 1
    assert response.choices[0].message.content == valid_response["choices"][0]["message"]["content"]
    assert response.model == valid_response["model"]

    # Test invalid object type
    with pytest.raises(ValidationError) as exc_info:
        CompletionsResponse(**{**valid_response, "object": "invalid_type"})
    assert "chat.completion" in str(exc_info.value)


def test_standard_inference_sampling_parameters():
    """Test validation of sampling parameters."""
    # Test with valid sampling parameters
    valid_params = {
        "messages": [{"role": "user", "content": "test"}],
        "sampling_parameters": {"temperature": 0.7, "top_p": 0.95, "do_sample": True, "max_new_tokens": 256},
    }
    request = CompletionsRequest(**valid_params)
    assert request.sampling_parameters["temperature"] == 0.7
    assert request.sampling_parameters["top_p"] == 0.95

    # Test with invalid temperature
    with pytest.raises(ValueError) as exc_info:
        CompletionsRequest(messages=[{"role": "user", "content": "test"}], sampling_parameters={"temperature": 2.0})
    assert "Temperature must be between 0 and 1" in str(exc_info.value)


def test_standard_inference_streaming_format():
    """Test streaming response format validation."""
    stream_chunk = {
        "id": "1dd0f0dc-bbc9-4be3-b143-c5340672e77e",
        "choices": [{"delta": {"content": "The Japanese", "role": "assistant"}, "finish_reason": None, "index": 0}],
        "created": 1741960995,
        "model": "hugging-quants/Meta-Llama-3.1-70B-Instruct-AWQ-INT4",
        "object": "chat.completion.chunk",
    }

    response = CompletionsResponse(**stream_chunk)
    assert response.object == "chat.completion.chunk"
    assert response.choices[0].delta.content == "The Japanese"


def test_standard_inference_message_validation():
    """Test validation of message objects."""
    # Test valid message
    valid_message = ChatCompletionMessage(role="assistant", content="Test content")
    assert valid_message.role == "assistant"
    assert valid_message.content == "Test content"

    # Test invalid role
    with pytest.raises(ValidationError) as exc_info:
        ChatCompletionMessage(role="invalid_role", content="Test content")
    assert "assistant" in str(exc_info.value)


def test_standard_inference_choice_validation():
    """Test validation of choice objects."""
    valid_choice = Choice(
        finish_reason="stop", index=0, message=ChatCompletionMessage(role="assistant", content="Test content")
    )
    assert valid_choice.finish_reason == "stop"
    assert valid_choice.index == 0
    assert valid_choice.message.content == "Test content"

    # Test invalid finish reason
    with pytest.raises(ValidationError) as exc_info:
        Choice(
            finish_reason="invalid_reason",
            index=0,
            message=ChatCompletionMessage(role="assistant", content="Test content"),
        )
    assert "stop" in str(exc_info.value)
