from fastapi import APIRouter, Request
import openai
from prompting.settings import settings
from httpx import Timeout
from prompting.base.epistula import create_header_hook
from fastapi.responses import StreamingResponse
import json
from prompting.miner_availability.miner_availability import miner_availabilities

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
    if not settings.mode == "mock" and not (
        available_miners := miner_availabilities.get_available_miners(task="Inference", model=None)
    ):
        return "No miners available"
    axon_info = settings.METAGRAPH.axons[available_miners[0]]
    base_url = "http://localhost:8008/v1" if settings.mode == "mock" else f"http://{axon_info.ip}:{axon_info.port}/v1"

    # TODO: Forward to actual miners
    miner = openai.AsyncOpenAI(
        base_url=base_url,
        max_retries=0,
        timeout=Timeout(settings.NEURON_TIMEOUT, connect=5, read=5),
        http_client=openai.DefaultAsyncHttpxClient(
            event_hooks={"request": [create_header_hook(settings.WALLET.hotkey, None)]}
        ),
    )

    # Create streaming request to OpenAI
    response = await miner.chat.completions.create(**body)

    # TODO: Add final response to scoring_queue

    # Return a streaming response with properly formatted chunks
    return StreamingResponse(process_stream(response), media_type="text/event-stream")
