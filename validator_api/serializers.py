from typing import List

from pydantic import BaseModel


# Need to refactor throughout codebase to use this type.
class Message(BaseModel):
    """Model for a message with role and content."""

    role: str
    content: str


class CompletionsRequest(BaseModel):
    """Request model for the /v1/chat/completions endpoint."""

    messages: List[Message]
    seed: int | None = None
    uids: List[int] | None = None
    task: str | None = None
    model: str | None = None
    test_time_inference: bool = False
    mixture: bool = False


class WebRetrievalRequest(BaseModel):
    """Request model for the /web_retrieval endpoint."""

    search_query: str
    n_miners: int = 10
    n_results: int = 5
    max_response_time: int = 10
    uids: List[int] | None = None


class WebSearchResult(BaseModel):
    """Model for a single web search results."""

    url: str
    content: str | None = None
    relevant: str | None = None


class WebRetrievalResponse(BaseModel):
    """Response model for the /web_retrieval endpoint."""

    results: List[WebSearchResult]


class TestTimeInferenceRequest(BaseModel):
    """Request model for the /test_time_inference endpoint."""

    messages: List[Message]
    model: str | None = None
    uids: List[int] | None = None
