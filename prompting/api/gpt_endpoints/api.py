from fastapi import APIRouter, Depends, Request

from prompting.api.api_managements.api import validate_api_key
from prompting.api.gpt_endpoints.mixture_of_miners import mixture_of_miners
from prompting.api.gpt_endpoints.process_completions import process_completions

router = APIRouter()


@router.post("/v1/chat/completions")
async def proxy_chat_completions(request: Request, api_key_data: dict = Depends(validate_api_key)):
    """OpenAI-style chat completions endpoint."""
    body = await request.json()
    if body.get("mixture", False):
        return await mixture_of_miners(body)
    else:
        return await process_completions(body)
