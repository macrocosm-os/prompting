import json

import pytest
from pydantic import ValidationError

from validator_api.serializers import WebRetrievalRequest, WebRetrievalResponse, WebSearchResult


def test_web_retrieval_request_validation():
    """Test that the request validation works correctly."""
    # Valid request
    valid_request = {
        "search_query": "Biggest fish in the world",
        "n_miners": 3,
        "n_results": 3,
        "max_response_time": 10,
    }
    request = WebRetrievalRequest(**valid_request)
    assert request.search_query == valid_request["search_query"]
    assert request.n_miners == valid_request["n_miners"]

    # Test required fields
    with pytest.raises(ValueError):
        WebRetrievalRequest()  # Missing required search_query

    # Test field constraints
    with pytest.raises(ValueError):
        WebRetrievalRequest(search_query="test", n_miners=0)  # Should be >= 1

    with pytest.raises(ValueError):
        WebRetrievalRequest(search_query="test", max_response_time=0)  # Should be >= 1


def test_web_retrieval_response_validation():
    """Test that the response validation works correctly."""
    valid_results = [
        WebSearchResult(
            url="https://example.com/whale-shark",
            content="The whale shark is the largest fish in the world",
            relevant="Direct answer about largest fish",
        ),
        WebSearchResult(
            url="https://example.com/basking-shark",
            content="The basking shark is the second largest fish",
            relevant=None,  # Optional field can be None
        ),
    ]

    response = WebRetrievalResponse(results=valid_results)
    assert len(response.results) == 2
    assert response.results[0].url == "https://example.com/whale-shark"

    # Test serialization
    response_dict = response.model_dump()
    assert isinstance(response_dict["results"], list)
    assert all(isinstance(r, dict) for r in response_dict["results"])


def test_web_search_result_content_types():
    """Test that the WebSearchResult handles different content types correctly."""
    # Test with actual content from the API response
    result = WebSearchResult(
        url="https://en.wikipedia.org/wiki/List_of_largest_fish",
        content="List of largest fish\nFish vary greatly in size...",
        relevant="List of largest fish\nFish vary greatly in size...",
    )

    # Verify content types
    assert isinstance(result.url, str)
    assert isinstance(result.content, str)
    assert isinstance(result.relevant, str)

    # Test with None values for optional fields
    result = WebSearchResult(url="https://example.com")
    assert result.content is None
    assert result.relevant is None


def test_web_retrieval_response_serialization():
    """Test that response serialization matches expected JSON structure."""
    response = WebRetrievalResponse(
        results=[
            WebSearchResult(
                url="https://en.wikipedia.org/wiki/List_of_largest_fish",
                content="List of largest fish\nFish vary greatly in size...",
                relevant="List of largest fish\nFish vary greatly in size...",
            )
        ]
    )

    serialized = response.model_dump()
    assert "results" in serialized
    assert len(serialized["results"]) == 1
    assert serialized["results"][0]["url"] == "https://en.wikipedia.org/wiki/List_of_largest_fish"
    assert serialized["results"][0]["content"].startswith("List of largest fish")


def test_web_retrieval_long_content():
    """Test that the response handles long content correctly."""
    long_content = "List of largest fish\n" * 100  # Create long content

    result = WebSearchResult(
        url="https://example.com", content=long_content, relevant=long_content[:100]  # Shorter relevant section
    )

    assert len(result.content) > 1000  # Verify long content is preserved
    assert len(result.relevant) == 100  # Verify relevant section length


def test_web_retrieval_special_characters():
    """Test that the response handles special characters in content."""
    content_with_special_chars = """
    Fish sizes:
    • Whale shark: 18.8 metres (61.7 ft)
    • Temperature: 25°C
    • Quote: "largest fish"
    • Em dash — separator
    """

    result = WebSearchResult(url="https://example.com", content=content_with_special_chars)

    serialized = result.model_dump_json()
    deserialized = json.loads(serialized)
    assert deserialized["content"] == content_with_special_chars


def test_web_retrieval_empty_results():
    """Test handling of empty results list."""
    response = WebRetrievalResponse(results=[])
    assert len(response.results) == 0

    serialized = response.model_dump()
    assert serialized["results"] == []


def test_web_retrieval_multiple_results():
    """Test handling of multiple results."""
    results = [
        WebSearchResult(url=f"https://example.com/{i}", content=f"Content {i}", relevant=f"Relevant {i}")
        for i in range(3)
    ]

    response = WebRetrievalResponse(results=results)
    assert len(response.results) == 3

    serialized = response.model_dump()
    assert len(serialized["results"]) == 3
    assert all(r["url"].startswith("https://example.com/") for r in serialized["results"])


def test_web_retrieval_result_deduplication():
    """Test that the response model can handle duplicate URLs (deduplication is handled at the API level)."""
    # Create test results with duplicate URLs
    duplicate_results = [
        WebSearchResult(url="https://example.com/fish", content="Content 1"),
        WebSearchResult(url="https://example.com/fish", content="Content 2"),  # Same URL
        WebSearchResult(url="https://example.com/shark", content="Different URL"),
    ]

    response = WebRetrievalResponse(results=duplicate_results)
    response_dict = response.model_dump()

    # Verify that the model preserves all results (deduplication happens in the API endpoint)
    urls = [r["url"] for r in response_dict["results"]]
    assert len(urls) == 3  # All results should be preserved
    assert urls.count("https://example.com/fish") == 2  # Both duplicate URLs should be present
    assert urls.count("https://example.com/shark") == 1  # Single URL should be present once


def test_web_retrieval_invalid_url():
    """Test validation of invalid URLs."""
    invalid_urls = [
        "",  # Empty string
        "not_a_url",  # No protocol
        "http:/",  # Incomplete protocol
        None,  # None value
    ]

    for url in invalid_urls:
        if url is None:
            with pytest.raises(ValidationError):
                WebSearchResult(url=url)
        else:
            # Note: Currently URL format is not strictly validated
            result = WebSearchResult(url=url)
            assert result.url == url
