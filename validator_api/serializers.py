from typing import List

from pydantic import BaseModel, Field


class WebSearchQuery(BaseModel):
    """Request model for web search queries."""

    search_query: str = Field(
        ...,
        description="The search query to execute using DuckDuckGo",
        example="latest developments in quantum computing",
    )
    n_miners: int = Field(
        default=10, description="Number of miners to query for results (between 1-100)", example=10, ge=1, le=100
    )
    uids: List[int] | None = Field(
        default=None,
        description="List of specific miner UIDs to query. If not provided, miners will be automatically selected.",
        example=[1, 2, 3, 4],
    )
    n_results: int = Field(
        default=5, description="Number of results each miner should return (between 1-30)", example=5, ge=1, le=30
    )

    class Config:
        schema_extra = {
            "example": {
                "search_query": "latest developments in quantum computing",
                "n_miners": 10,
                "uids": [1, 2, 3, 4],
                "n_results": 5,
            }
        }


class SearchResult(BaseModel):
    """Model representing a single search result."""

    url: str = Field(
        ..., description="URL of the search result", example="https://www.nature.com/articles/d41586-023-02192-x"
    )
    title: str = Field(
        ...,
        description="Title of the webpage or document",
        example="Quantum computing breakthrough: New superconducting qubits",
    )
    snippet: str = Field(
        ...,
        description="Brief excerpt or summary of the content",
        example="Researchers have developed a new type of superconducting qubit that increases stability by 50%.",
    )
    timestamp: str = Field(..., description="Timestamp when the result was retrieved", example="2024-01-01 12:00:00")


class WebSearchResponse(BaseModel):
    """Response model containing search results from distributed miners."""

    results: List[SearchResult] = Field(..., description="List of deduplicated and parsed search results")

    class Config:
        schema_extra = {
            "example": {
                "results": [
                    {
                        "url": "https://www.nature.com/articles/d41586-023-02192-x",
                        "title": "Quantum computing breakthrough: New superconducting qubits",
                        "snippet": "Researchers have developed a new type of superconducting qubit that increases stability by 50%.",
                        "timestamp": "2024-01-01 12:00:00",
                    },
                    {
                        "url": "https://arxiv.org/abs/2307.05230",
                        "title": "Arxiv paper: Enhancing Quantum Error Correction",
                        "snippet": "A novel quantum error correction technique reduces noise in superconducting circuits.",
                        "timestamp": "2024-01-02 09:30:00",
                    },
                ]
            }
        }


class ErrorResponse(BaseModel):
    """Model for API error responses."""

    detail: str = Field(
        ...,
        description="Detailed error message explaining what went wrong",
        example="No available miners found to process the request",
    )


class ChatMessage(BaseModel):
    """Model representing a single chat message."""

    role: str = Field(
        ..., description="Role of the message sender", example="user", enum=["user", "assistant", "system"]
    )
    content: str = Field(..., description="The message content", example="What is the meaning of life?")


class ChatCompletionRequest(BaseModel):
    """Request model for chat completion."""

    messages: List[dict] = Field(
        ...,
        description="List of chat messages containing user input and system responses",
        example=[
            {"role": "user", "content": "What is the meaning of life?"},
            {"role": "assistant", "content": "Philosophers have debated this question for centuries."},
        ],
    )
    model: str = Field(
        default="gpt-4",
        description="Model to use for generating responses",
        example="gpt-4",
        enum=["gpt-4", "gpt-3.5-turbo", "custom-model"],
    )
    seed: int | None = Field(
        default=None,
        description="Random seed for response generation. If not provided, a random seed will be generated.",
        example=42,
    )
    test_time_inference: bool = Field(
        default=False, description="When true, enables test-time inference mode", example=False
    )
    mixture: bool = Field(default=False, description="When true, enables mixture-of-miners strategy", example=False)
    uids: List[int] | None = Field(
        default=None,
        title="Miner UIDs",
        description="List of specific miner UIDs to query. If not provided, miners will be automatically selected.",
        example=[1, 5, 7],
    )

    class Config:
        schema_extra = {
            "example": {
                "messages": [{"role": "user", "content": "What is quantum computing?"}],
                "model": "gpt-4",
                "seed": 42,
                "test_time_inference": False,
                "mixture": False,
                "uids": [1, 5, 7],
            }
        }
