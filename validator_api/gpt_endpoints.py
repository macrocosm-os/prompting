import asyncio
import json
import random

import httpx
from fastapi import APIRouter, HTTPException, Request
from loguru import logger
from starlette.responses import StreamingResponse

from shared.epistula import make_openai_query
from shared.settings import shared_settings

router = APIRouter()


async def forward_response(uid: int, body: dict[str, any], chunks: list[str]):
    if not shared_settings.SCORE_ORGANICS:  # Allow disabling of scoring by default
        return

    # if body.get("task") != "InferenceTask":
    #     logger.info(f"Skipping forwarding for non-inference task: {body.get('task')}")
    #     return
    url = f"http://{shared_settings.VALIDATOR_API}/scoring"
    logger.info(url)
    payload = {"body": body, "chunks": chunks, "uid": uid}
    # headers = {
    #     "Authorization": f"Bearer {shared_settings.SCORING_KEY}",  #Add API key in Authorization header
    #     "Content-Type": "application/json",
    # }
    try:
        timeout = httpx.Timeout(timeout=120.0, connect=60.0, read=30.0, write=30.0, pool=5.0)
        async with httpx.AsyncClient(timeout=timeout) as client:
            logger.debug(f"Payload: {payload}")
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


@router.post("/v1/chat/completions")
async def chat_completion(request: Request):  # , cbackground_tasks: BackgroundTasks):
    try:
        body = await request.json()
        body["seed"] = int(body.get("seed") or random.randint(0, 1000000))
        STREAM = body.get("stream") or False
        logger.debug(f"Streaming: {STREAM}")
        uid = random.randint(0, len(shared_settings.METAGRAPH.axons) - 1)
        # uid = get_available_miner(task=body.get("task"), model=body.get("model"))
        if uid is None:
            logger.error("No available miner found")
            raise HTTPException(status_code=503, detail="No available miner found")
        logger.debug(f"Querying uid {uid}")

        collected_chunks: list[str] = []

        # Create a wrapper for the streaming response
        async def stream_with_error_handling():
            try:
                async for chunk in response:
                    collected_chunks.append(chunk.choices[0].delta.content)
                    yield f"data: {json.dumps(chunk.model_dump())}\n\n"
                yield "data: [DONE]\n\n"
                # Once the stream is done, forward the collected chunks
                asyncio.create_task(forward_response(uid=uid, body=body, chunks=collected_chunks))
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
