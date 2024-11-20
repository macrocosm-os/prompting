from fastapi import APIRouter, Request
import openai
from prompting.settings import settings
from httpx import Timeout
from prompting.base.epistula import create_header_hook
from fastapi.responses import StreamingResponse
import json

router = APIRouter()


async def process_stream(stream):
    async for chunk in stream:
        if hasattr(chunk, "choices") and chunk.choices:
            # Extract the delta content from the chunk
            delta = chunk.choices[0].delta
            if hasattr(delta, "content") and delta.content is not None:
                # Format as SSE data
                yield f"data: {json.dumps(chunk.model_dump())}\n\n"
    yield "data: [DONE]\n\n"


@router.post("/v1/chat/completions")
async def proxy_chat_completions(request: Request):
    # Get the request body
    body = await request.json()

    # Ensure streaming is enabled
    body["stream"] = True

    # TODO: Forward to actual miners
    miner = openai.AsyncOpenAI(
        base_url="http://localhost:8008/v1",
        max_retries=0,
        timeout=Timeout(settings.NEURON_TIMEOUT, connect=5, read=5),
        http_client=openai.DefaultAsyncHttpxClient(
            event_hooks={"request": [create_header_hook(settings.WALLET.hotkey, None)]}
        ),
    )

    # Create streaming request to OpenAI
    response = await miner.chat.completions.create(**body)

    # Return a streaming response with properly formatted chunks
    return StreamingResponse(process_stream(response), media_type="text/event-stream")
