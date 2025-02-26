from typing import List

from pydantic import BaseModel


class CompletionsRequest(BaseModel):
    """Request model for the /v1/chat/completions endpoint."""

    uids: List[int] | None = None
    messages: List[dict[str, str]]
    seed: int | None = None
    task: str | None = None
    model: str | None = None
    test_time_inference: bool = False
    mixture: bool = False
    sampling_parameters: dict | None = None




class WebRetrievalRequest(BaseModel):
    """Request model for the /web_retrieval endpoint."""

    uids: List[int] | None = None
    search_query: str
    n_miners: int = 10
    n_results: int = 5
    max_response_time: int = 10


class WebSearchResult(BaseModel):
    """Model for a single web search results."""

    url: str
    content: str | None = None
    relevant: str | None = None


class WebRetrievalResponse(BaseModel):
    """Response model for the /web_retrieval endpoint."""

    results: List[WebSearchResult]

    def to_dict(self):
        return self.model_dump().update({"results": [r.model_dump() for r in self.results]})


class TestTimeInferenceRequest(BaseModel):
    """Request model for the /test_time_inference endpoint."""

    uids: List[int] | None = None
    messages: List[dict[str, str]]
    model: str | None = None

    def to_dict(self):
        return self.model_dump().update({"messages": [m.model_dump() for m in self.messages]})
