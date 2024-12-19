import random

from fastapi import APIRouter, Request
from loguru import logger
from starlette.responses import StreamingResponse

from validator_api.mixture_of_miners import mixture_of_miners
from validator_api.chat_completion import chat_completion

router = APIRouter()


@router.post("/v1/chat/completions")
async def completions(request: Request):
    """Main endpoint that handles both regular and mixture of miners chat completion."""
    try:
        body = await request.json()
        body["seed"] = int(body.get("seed") or random.randint(0, 1000000))
        
        # Choose between regular completion and mixture of miners.
        if body.get("mixture", False):
            return await mixture_of_miners(body)
        else:
            return await chat_completion(body)

    except Exception as e:
        logger.exception(f"Error in chat completion: {e}")
        return StreamingResponse(content="Internal Server Error", status_code=500)
