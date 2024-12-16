import asyncio
import json
import random

from fastapi import APIRouter, Request
from loguru import logger
from starlette.responses import StreamingResponse

from shared.epistula import make_openai_query
from shared.settings import shared_settings

router = APIRouter()


@router.post("/v1/chat/completions")
async def chat_completion(request: Request):
    try:
        body = await request.json()
        STREAM = body.get("stream") or False
        logger.debug(f"Streaming: {STREAM}")
        uid = random.randint(0, len(shared_settings.METAGRAPH.axons) - 1)
        logger.debug(f"Querying uid {uid}")

        # Create a wrapper for the streaming response
        async def stream_with_error_handling():
            try:
                async for chunk in response:
                    logger.debug(chunk)
                    yield f"data: {json.dumps(chunk.model_dump())}\n\n"
                yield "data: [DONE]\n\n"
            except asyncio.CancelledError:
                logger.info("Client disconnected, streaming cancelled")
                raise
            except Exception as e:
                logger.exception(f"Error during streaming: {e}")
                yield 'data: {"error": "Internal server Error"}\n\n'

        logger.info(f"Making {'streaming' if STREAM else 'non-streaming'} openai query with body: {body}")
        response = await make_openai_query(shared_settings.METAGRAPH, shared_settings.WALLET, body, uid, stream=STREAM)
        if STREAM:
            return StreamingResponse(
                stream_with_error_handling(),
                media_type="text/event-stream",
                headers={
                    "Cache-Control": "no-cache",
                    "Connection": "keep-alive",
                },
            )
        else:
            return response[0]

    except Exception as e:
        logger.exception(f"Error setting up streaming: {e}")
        return StreamingResponse(content="Internal Server Error", status_code=500)
