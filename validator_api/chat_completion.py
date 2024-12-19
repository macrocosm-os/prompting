import asyncio
import json
import random
from typing import AsyncGenerator

import httpx
from fastapi import HTTPException
from fastapi.responses import StreamingResponse
from loguru import logger

from shared.epistula import make_openai_query
from shared.settings import shared_settings
from shared.uids import get_uids


async def forward_response(uid: int, body: dict[str, any], chunks: list[str]):
    if not shared_settings.SCORE_ORGANICS:  # Allow disabling of scoring by default
        return

    # if body.get("task") != "InferenceTask":
    #     logger.debug(f"Skipping forwarding for non-inference task: {body.get('task')}")
    #     return
    url = f"http://{shared_settings.VALIDATOR_API}/scoring"
    payload = {"body": body, "chunks": chunks, "uid": uid}
    # headers = {
    #     "Authorization": f"Bearer {shared_settings.SCORING_KEY}",  #Add API key in Authorization header
    #     "Content-Type": "application/json",
    # }
    try:
        timeout = httpx.Timeout(timeout=120.0, connect=60.0, read=30.0, write=30.0, pool=5.0)
        async with httpx.AsyncClient(timeout=timeout) as client:
            response = await client.post(url, json=payload)  # , headers=headers)
            if response.status_code == 200:
                logger.info(f"Forwarding response completed with status {response.status_code}")

            else:
                logger.exception(
                    f"Forwarding response uid {uid} failed with status {response.status_code} and payload {payload}"
                )

    except Exception as e:
        logger.error(f"Tried to forward response to {url} with payload {payload}")
        logger.exception(f"Error while forwarding response: {e}")


async def stream_response(
    response, collected_chunks: list[str], body: dict[str, any], uid: int
) -> AsyncGenerator[str, None]:
    chunks_received = False
    try:
        async for chunk in response:
            chunks_received = True
            collected_chunks.append(chunk.choices[0].delta.content)
            yield f"data: {json.dumps(chunk.model_dump())}\n\n"

        if not chunks_received:
            logger.error("Stream is empty: No chunks were received")
            yield 'data: {"error": "502 - Response is empty"}\n\n'
        yield "data: [DONE]\n\n"

        # Forward the collected chunks after streaming is complete
        asyncio.create_task(forward_response(uid=uid, body=body, chunks=collected_chunks))
    except asyncio.CancelledError:
        logger.info("Client disconnected, streaming cancelled")
        raise
    except Exception as e:
        logger.exception(f"Error during streaming: {e}")
        yield 'data: {"error": "Internal server Error"}\n\n'


async def chat_completion(body: dict[str, any], uid: int | None = None) -> tuple | StreamingResponse:
    """Handle regular chat completion without mixture of miners."""
    if uid is None:
        uid = random.choice(get_uids(sampling_mode="top_incentive", k=100))

    if uid is None:
        logger.error("No available miner found")
        raise HTTPException(status_code=503, detail="No available miner found")

    logger.debug(f"Querying uid {uid}")
    STREAM = body.get("stream", False)

    collected_chunks: list[str] = []

    logger.info(f"Making {'streaming' if STREAM else 'non-streaming'} openai query with body: {body}")
    response = await make_openai_query(shared_settings.METAGRAPH, shared_settings.WALLET, body, uid, stream=STREAM)

    if STREAM:
        return StreamingResponse(
            stream_response(response, collected_chunks, body, uid),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
            },
        )
    else:
        asyncio.create_task(forward_response(uid=uid, body=body, chunks=response[1]))
        return response[0]


async def get_response_from_miner(body: dict[str, any], uid: int) -> tuple:
    """Get response from a single miner."""
    return await make_openai_query(shared_settings.METAGRAPH, shared_settings.WALLET, body, uid, stream=False)
