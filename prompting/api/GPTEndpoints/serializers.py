from pydantic import BaseModel


class ChatCompletionRequest(BaseModel):
    messages: list[dict]


class ChatCompletionResponse(BaseModel):
    messages: list[dict]
    response: str
    created: int
    choices: list[dict]
