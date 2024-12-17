import asyncio
import json

import httpx
from fastapi import APIRouter, HTTPException, Request
from loguru import logger
from starlette.responses import StreamingResponse

from shared.epistula import make_openai_query
from shared.settings import shared_settings
from validator_api.miner_availabilities import get_available_miner

router = APIRouter()


async def forward_response(uid: int, body: dict[str, any], chunks: list[str]):
    if body.get("task") != "InferenceTask":
        logger.info(f"Skipping forwarding for non-inference task: {body.get('task')}")
        return
    url = f"http://{shared_settings.VALIDATOR_IP}:{shared_settings.VALIDATOR_PORT}/scoring"
    payload = {"body": body, "response": chunks, "uid": uid}
    headers = {
        "Authorization": f"Bearer {shared_settings.SCORING_KEY}",  # Add API key in Authorization header
        "Content-Type": "application/json",
    }
    try:
        async with httpx.AsyncClient() as client:
            response = await client.post(url, json=payload, headers=headers)
            logger.info(f"Forwarding response completed with status {response.status_code}")
    except Exception as e:
        logger.error(f"Tried to forward response to {url} with payload {payload} and headers {headers}")
        logger.exception(f"Error while forwarding response: {e}")


@router.post("/v1/chat/completions")
async def chat_completion(request: Request):  # , cbackground_tasks: BackgroundTasks):
    try:
        body = await request.json()
        STREAM = body.get("stream") or False
        logger.debug(f"Streaming: {STREAM}")
        # uid = random.randint(0, len(shared_settings.METAGRAPH.axons) - 1)
        uid = get_available_miner(task=body.get("task"), model=body.get("model"))
        if uid is None:
            logger.error("No available miner found")
            raise HTTPException(status_code=503, detail="No available miner found")
        logger.debug(f"Querying uid {uid}")

        collected_chunks: list[str] = []

        # Create a wrapper for the streaming response
        async def stream_with_error_handling():
            try:
                async for chunk in response:
                    logger.debug(chunk)
                    collected_chunks.append(chunk.model_dump())
                    yield f"data: {json.dumps(chunk.model_dump())}\n\n"
                yield "data: [DONE]\n\n"
                # Once the stream is done, forward the collected chunks
                # asyncio.create_task(forward_response(uid=uid, body=body, chunks=collected_chunks))
                # background_tasks.add_task(forward_response, uid=uid, body=body, chunks=collected_chunks)
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
            asyncio.create_task(forward_response(uid=uid, body=body, chunks=response[1]))
            return response[0]

    except Exception as e:
        logger.exception(f"Error setting up streaming: {e}")
        return StreamingResponse(content="Internal Server Error", status_code=500)
