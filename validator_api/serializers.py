from pydantic import BaseModel, Field
from typing import Optional

class WebSearchQuery(BaseModel):
    """Request model for web search queries."""
    search_query: str = Field(
        ...,
        description="The search query to be executed using DuckDuckGo",
        example="latest developments in quantum computing"
    )
    n_miners: int = Field(
        default=10,
        ge=1,
        le=100,
        description="Number of miners to query for results",
        example=10
    )
    uids: Optional[list[int]] = Field(
        default=None,
        description="Specific miner UIDs to query. If not provided, available miners will be automatically selected",
        example=[1, 2, 3, 4]
    )
    n_results: int = Field(
        default=5,
        ge=1,
        le=30,
        description="Number of results each miner should return",
        example=5
    )

class SearchResult(BaseModel):
    """Model representing a single search result."""
    url: str = Field(..., example="https://www.google.com", description="URL of the search result")
    title: str = Field(..., example="Google", description="Title of the webpage or document")
    snippet: str = Field(..., example="Google is a search engine", description="Brief excerpt or summary of the content")
    timestamp: str = Field(..., example="2024-01-01 12:00:00", description="Timestamp of when the result was retrieved")

class WebSearchResponse(BaseModel):
    """Response model for web search results."""
    results: list[SearchResult] = Field(
        ...,
        description="list of search results from different miners"
    )

class ErrorResponse(BaseModel):
    """Model for error responses."""
    detail: str = Field(..., description="Detailed error message")
