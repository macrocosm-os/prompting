from fastapi import APIRouter
from prompting.api.GPTEndpoints.serializers import ChatCompletionRequest, ChatCompletionResponse

router = APIRouter()


@router.post("/")
async def chat_completion(request: ChatCompletionRequest):
    return ChatCompletionResponse(
        messages=request.messages,
        response="Boop",
        created=1234567890,
        choices=[],
    )
