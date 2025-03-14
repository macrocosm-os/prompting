from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field, field_validator


class CompletionsRequest(BaseModel):
    """Request model for the /v1/chat/completions endpoint."""

    uids: Optional[List[int]] = Field(
        default=None,
        description="List of specific miner UIDs to query. If not provided, miners will be selected automatically.",
        json_schema_extra={"example": [1, 2, 3]},
    )
    messages: List[Dict[str, str]] = Field(
        ...,
        description="List of message objects with 'role' and 'content' keys. Roles can be 'system', 'user', or 'assistant'.",
        json_schema_extra={"example": [{"role": "user", "content": "Tell me about neural networks"}]},
    )
    seed: Optional[int] = Field(
        default=None,
        description="Random seed for reproducible results. If not provided, a random seed will be generated.",
        json_schema_extra={"example": 42},
    )
    task: Optional[str] = Field(
        default="InferenceTask",
        description="Task identifier to choose the inference type.",
        json_schema_extra={"example": "InferenceTask"},
    )
    model: Optional[str] = Field(
        default=None,
        description="Model identifier to filter available miners.",
        json_schema_extra={"example": "hugging-quants/Meta-Llama-3.1-70B-Instruct-AWQ-INT4"},
    )
    test_time_inference: bool = Field(
        default=False, description="Enable step-by-step reasoning mode that shows the model's thinking process."
    )
    mixture: bool = Field(
        default=False, description="Enable mixture of miners mode that combines responses from multiple miners."
    )
    sampling_parameters: Optional[Dict[str, Any]] = Field(
        default={
            "temperature": 0.7,
            "top_p": 0.95,
            "top_k": 50,
            "max_new_tokens": 1024,
            "do_sample": True,
        },
        description="Parameters to control text generation, such as temperature, top_p, etc.",
        json_schema_extra={
            "example": {
                "temperature": 0.7,
                "top_p": 0.95,
                "top_k": 50,
                "max_new_tokens": 512,
                "do_sample": True,
            }
        },
    )
    inference_mode: Optional[str] = Field(
        default=None,
        description="Inference mode to use for the task.",
        json_schema_extra={"example": "Reasoning-Fast"},
    )
    json_format: bool = Field(
        default=False,
        description="Enable JSON format for the response.",
        json_schema_extra={"example": True},
    )
    stream: bool = Field(
        default=False,
        description="Enable streaming for the response.",
        json_schema_extra={"example": True},
    )

    @field_validator("messages")
    def validate_messages(cls, v):
        for msg in v:
            if "role" not in msg or "content" not in msg:
                raise ValueError("Each message must contain 'role' and 'content' keys")
            if msg["role"] not in ["system", "user", "assistant"]:
                raise ValueError("Message role must be one of: system, user, assistant")
        return v

    @field_validator("sampling_parameters")
    def validate_sampling_parameters(cls, v):
        if v is None:
            return v
        if "temperature" in v and (v["temperature"] < 0 or v["temperature"] > 1):
            raise ValueError("Temperature must be between 0 and 1")
        if "top_p" in v and (v["top_p"] < 0 or v["top_p"] > 1):
            raise ValueError("Top_p must be between 0 and 1")
        return v


class WebRetrievalRequest(BaseModel):
    """Request model for the /web_retrieval endpoint."""

    uids: Optional[List[int]] = Field(
        default=None,
        description="List of specific miner UIDs to query. If not provided, miners will be selected automatically.",
        json_schema_extra={"example": [1, 2, 3]},
    )
    search_query: str = Field(
        ...,
        description="The query to search for on the web.",
        json_schema_extra={"example": "latest advancements in quantum computing"},
    )
    n_miners: int = Field(
        default=3,
        description="Number of miners to query for results.",
        json_schema_extra={"example": 15},
        ge=1,
    )
    n_results: int = Field(
        default=1,
        description="Maximum number of results to return in the response.",
        json_schema_extra={"example": 5},
        ge=1,
    )
    max_response_time: int = Field(
        default=10,
        description="Maximum time to wait for responses in seconds.",
        json_schema_extra={"example": 15},
        ge=1,
    )


class WebSearchResult(BaseModel):
    """Model for a single web search result."""

    url: str = Field(
        ...,
        description="The URL of the web page.",
        json_schema_extra={"example": "https://example.com/article"},
    )
    content: Optional[str] = Field(
        default=None,
        description="The relevant content extracted from the page.",
        json_schema_extra={"example": "Quantum computing has seen significant advancements in the past year..."},
    )
    relevant: Optional[str] = Field(
        default=None,
        description="Information about why this result is relevant to the query.",
        json_schema_extra={"example": "This article discusses the latest breakthroughs in quantum computing research."},
    )


class WebRetrievalResponse(BaseModel):
    """Response model for the /web_retrieval endpoint."""

    results: List[WebSearchResult] = Field(..., description="List of unique web search results.")

    def to_dict(self):
        return self.model_dump().update({"results": [r.model_dump() for r in self.results]})


class InferenceRequest(BaseModel):
    """Request model for the /test_time_inference endpoint."""

    uids: Optional[List[int]] = Field(
        default=None,
        description="List of specific miner UIDs to query. If not provided, miners will be selected automatically.",
        json_schema_extra={"example": [1, 2, 3]},
    )
    messages: List[Dict[str, str]] = Field(
        ...,
        description="List of message objects with 'role' and 'content' keys. Roles can be 'system', 'user', or 'assistant'.",
        json_schema_extra={"example": [{"role": "user", "content": "Solve the equation: 3x + 5 = 14"}]},
    )
    model: Optional[str] = Field(
        default=None,
        description="Model identifier to use for inference.",
        json_schema_extra={"example": "gpt-4"},
    )
    json_format: bool = Field(
        default=False,
        description="Enable JSON format for the response.",
        json_schema_extra={"example": True},
    )

    def to_dict(self):
        return self.model_dump().update({"messages": [m.model_dump() for m in self.messages]})


class ChatCompletionMessage(BaseModel):
    """Model for chat completion message."""

    role: Literal["system", "user", "assistant"] = Field(
        ..., description="Role of the message sender (system, user, or assistant)"
    )
    content: Optional[str] = Field(None, description="Content of the message")
    refusal: Optional[str] = Field(None, description="Refusal message if applicable")
    audio: Optional[Any] = Field(None, description="Audio content if applicable")
    function_call: Optional[Any] = Field(None, description="Function call details if applicable")
    tool_calls: Optional[Any] = Field(None, description="Tool call details if applicable")


class ChatCompletionDelta(BaseModel):
    """Model for streaming delta updates."""

    role: Optional[Literal["system", "user", "assistant"]] = None
    content: Optional[str] = None


class Choice(BaseModel):
    """Model for completion choices."""

    finish_reason: Optional[Literal["stop", "length", "content_filter", "function_call", "tool_calls"]] = Field(
        None, description="Reason for finishing (stop, length, etc)"
    )
    index: int = Field(..., description="Index of the choice", ge=0)
    logprobs: Optional[Any] = Field(None, description="Log probabilities if requested")
    message: Optional[ChatCompletionMessage] = Field(None, description="The message for non-streaming responses")
    delta: Optional[ChatCompletionDelta] = Field(None, description="The delta for streaming responses")


class CompletionsResponse(BaseModel):
    """Response model for the /v1/chat/completions endpoint."""

    id: str = Field(..., description="Unique identifier for the completion")
    choices: List[Choice] = Field(..., description="List of completion choices")
    created: int = Field(..., description="Unix timestamp of when the completion was created")
    model: str = Field(..., description="Model used for the completion")
    object: Literal["chat.completion", "chat.completion.chunk"] = Field(
        ..., description="Object type (chat.completion or chat.completion.chunk)"
    )
    service_tier: Optional[str] = Field(None, description="Service tier information if applicable")
    system_fingerprint: Optional[str] = Field(None, description="System fingerprint if applicable")
    usage: Optional[Dict[str, int]] = Field(None, description="Token usage information if available")
