import json

import pytest
from pydantic import ValidationError

from validator_api.serializers import (
    CompletionsRequest,
    InferenceRequest,
    WebRetrievalRequest,
    WebRetrievalResponse,
    WebSearchResult,
)

# Snapshot of expected JSON schema - if this changes, frontend needs to be notified
EXPECTED_COMPLETIONS_SCHEMA = {
    "type": "object",
    "properties": {
        "messages": {"type": "array", "items": {"additionalProperties": {"type": "string"}, "type": "object"}},
        "uids": {"anyOf": [{"type": "array", "items": {"type": "integer"}}, {"type": "null"}], "default": None},
        "seed": {"anyOf": [{"type": "integer"}, {"type": "null"}], "default": None},
        "task": {"anyOf": [{"type": "string"}, {"type": "null"}], "default": "InferenceTask"},
        "model": {"anyOf": [{"type": "string"}, {"type": "null"}], "default": None},
        "test_time_inference": {"type": "boolean", "default": False},
        "mixture": {"type": "boolean", "default": False},
        "sampling_parameters": {
            "anyOf": [{"type": "object"}, {"type": "null"}],
            "default": {"temperature": 0.7, "top_p": 0.95, "top_k": 50, "max_new_tokens": 1024, "do_sample": True},
        },
        "inference_mode": {"anyOf": [{"type": "string"}, {"type": "null"}], "default": None},
        "json_format": {"type": "boolean", "default": False},
        "stream": {"type": "boolean", "default": False},
    },
    "required": ["messages"],
}


def test_completions_schema_matches_expected():
    """
    Test that ensures the CompletionsRequest schema hasn't changed in a way that would break the frontend.
    If this test fails, the frontend team needs to be notified of the schema changes.
    """
    schema = CompletionsRequest.model_json_schema()

    # Helper function to clean schema for comparison
    def clean_schema(s):
        """Remove metadata fields that don't affect the contract."""
        if isinstance(s, dict):
            return {
                k: clean_schema(v)
                for k, v in s.items()
                if k not in ["title", "description", "examples", "$defs", "example"]
            }
        return s

    cleaned_schema = clean_schema(schema)
    cleaned_expected = clean_schema(EXPECTED_COMPLETIONS_SCHEMA)

    try:
        assert cleaned_schema == cleaned_expected
    except AssertionError:
        print("\nBREAKING CHANGES DETECTED! Frontend team needs to be notified!")
        print("\nActual schema (cleaned):")
        print(json.dumps(cleaned_schema, indent=2))
        print("\nExpected schema (cleaned):")
        print(json.dumps(cleaned_expected, indent=2))
        raise


def test_completions_response_structure():
    """Test that the response structure matches what frontend expects."""
    valid_data = {"messages": [{"role": "user", "content": "Hello"}], "stream": True}

    request = CompletionsRequest(**valid_data)
    response_dict = request.model_dump()

    # Test that critical fields maintain their types
    assert isinstance(response_dict.get("messages"), list)
    assert isinstance(response_dict.get("stream"), bool)
    assert isinstance(response_dict.get("sampling_parameters", {}), dict)


def test_web_retrieval_request_schema():
    """Test WebRetrievalRequest schema stability."""
    schema = WebRetrievalRequest.model_json_schema()

    # Verify critical field types
    assert schema["properties"]["search_query"]["type"] == "string"
    assert schema["properties"]["n_miners"]["type"] == "integer"
    assert schema["required"] == ["search_query"]


def test_web_retrieval_response_schema():
    """Test WebRetrievalResponse schema stability."""
    schema = WebRetrievalResponse.model_json_schema()

    # Verify response structure
    assert "results" in schema["properties"]
    assert schema["properties"]["results"]["type"] == "array"


def test_serialization_format():
    """
    Test that serialization format remains consistent.
    Frontend often relies on specific JSON formatting.
    """
    request = CompletionsRequest(
        messages=[{"role": "user", "content": "Hello"}], sampling_parameters={"temperature": 0.7}
    )

    serialized = request.model_dump_json()
    deserialized = json.loads(serialized)

    # Verify JSON structure
    assert isinstance(deserialized, dict)
    assert "messages" in deserialized
    assert isinstance(deserialized["messages"], list)
    assert isinstance(deserialized["sampling_parameters"], dict)


def test_field_constraints():
    """Test that field constraints remain consistent."""
    # Test n_miners constraints
    with pytest.raises(ValidationError):
        WebRetrievalRequest(search_query="test", n_miners=0)  # Should be >= 1

    # Test max_response_time constraints
    with pytest.raises(ValidationError):
        WebRetrievalRequest(search_query="test", max_response_time=0)  # Should be >= 1


def test_optional_fields_remain_optional():
    """
    Test that optional fields stay optional.
    Frontend might not always send all fields.
    """
    # Test each model with minimal required fields
    CompletionsRequest(messages=[{"role": "user", "content": "Hello"}])
    WebRetrievalRequest(search_query="test")
    WebSearchResult(url="https://example.com")
    InferenceRequest(messages=[{"role": "user", "content": "Hello"}])


def test_default_values_consistency():
    """
    Test that default values remain consistent.
    Frontend might rely on specific defaults.
    """
    request = CompletionsRequest(messages=[{"role": "user", "content": "Hello"}])

    assert request.sampling_parameters == {
        "temperature": 0.7,
        "top_p": 0.95,
        "top_k": 50,
        "max_new_tokens": 1024,
        "do_sample": True,
    }
    assert request.test_time_inference is False
    assert request.mixture is False


def test_completions_request_valid():
    """Test CompletionsRequest with valid data."""
    valid_data = {
        "messages": [{"role": "user", "content": "Hello"}],
        "uids": [1, 2, 3],
        "seed": 42,
        "task": "InferenceTask",
        "model": "hugging-quants/Meta-Llama-3.1-70B-Instruct-AWQ-INT4",
        "test_time_inference": False,
        "mixture": False,
        "sampling_parameters": {
            "temperature": 0.7,
            "top_p": 0.95,
            "top_k": 50,
            "max_new_tokens": 1024,
            "do_sample": True,
        },
        "inference_mode": "Reasoning-Fast",
        "json_format": False,
        "stream": True,
    }

    request = CompletionsRequest(**valid_data)
    assert request.model_dump() == valid_data


def test_completions_request_required_fields():
    """Test CompletionsRequest with only required fields."""
    minimal_data = {
        "messages": [{"role": "user", "content": "Hello"}],
    }

    request = CompletionsRequest(**minimal_data)
    assert request.messages == minimal_data["messages"]
    assert request.sampling_parameters == {
        "temperature": 0.7,
        "top_p": 0.95,
        "top_k": 50,
        "max_new_tokens": 1024,
        "do_sample": True,
    }


def test_completions_request_invalid():
    """Test CompletionsRequest with invalid data."""
    invalid_data = {
        "messages": "not_a_list",  # Should be a list
        "uids": ["not_integers"],  # Should be integers
        "seed": "not_an_integer",  # Should be an integer
    }

    with pytest.raises(ValidationError):
        CompletionsRequest(**invalid_data)


def test_web_retrieval_request_valid():
    """Test WebRetrievalRequest with valid data."""
    valid_data = {
        "uids": [1, 2, 3],
        "search_query": "quantum computing",
        "n_miners": 3,
        "n_results": 5,
        "max_response_time": 15,
    }

    request = WebRetrievalRequest(**valid_data)
    assert request.model_dump() == valid_data


def test_web_retrieval_response_valid():
    """Test WebRetrievalResponse with valid data."""
    results = [
        WebSearchResult(
            url="https://example.com",
            content="Sample content",
            relevant="Sample relevance",
        ),
        WebSearchResult(
            url="https://example.org",
            content="Another content",
        ),
    ]

    response = WebRetrievalResponse(results=results)
    assert len(response.results) == 2
    assert response.results[0].url == "https://example.com"
    assert response.results[1].content == "Another content"


def test_inference_request_valid():
    """Test inference request with valid data."""
    valid_data = {
        "uids": [1, 2, 3],
        "messages": [{"role": "user", "content": "Solve: 2x + 5 = 15"}],
        "model": "gpt-4",
        "json_format": True,
    }

    request = InferenceRequest(**valid_data)
    assert request.model_dump() == valid_data


def test_web_search_result_valid():
    """Test WebSearchResult with valid data."""
    valid_data = {
        "url": "https://example.com",
        "content": "Sample content",
        "relevant": "Sample relevance",
    }

    result = WebSearchResult(**valid_data)
    assert result.model_dump() == valid_data


def test_web_search_result_minimal():
    """Test WebSearchResult with only required fields."""
    minimal_data = {
        "url": "https://example.com",
    }

    result = WebSearchResult(**minimal_data)
    assert result.url == minimal_data["url"]
    assert result.content is None
    assert result.relevant is None
