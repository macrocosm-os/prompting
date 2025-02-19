import asyncio
import json
import random
import time
import uuid

import numpy as np
from fastapi import APIRouter, Depends, HTTPException, Request
from loguru import logger
from openai.types.chat.chat_completion_chunk import ChatCompletionChunk, Choice, ChoiceDelta
from starlette.responses import StreamingResponse

from shared import settings

shared_settings = settings.shared_settings
from shared.epistula import SynapseStreamResult, query_miners
from validator_api import scoring_queue
from validator_api.api_management import validate_api_key
from validator_api.chat_completion import chat_completion
from validator_api.mixture_of_miners import mixture_of_miners
from validator_api.test_time_inference import generate_response
from validator_api.utils import filter_available_uids

router = APIRouter()
N_MINERS = 5


@router.post("/v1/chat/completions")
async def completions(request: Request, api_key: str = Depends(validate_api_key)):
    """Main endpoint that handles both regular and mixture of miners chat completion."""
    try:
        body = await request.json()
        body["seed"] = int(body.get("seed") or random.randint(0, 1000000))
        if body.get("uids"):
            try:
                uids = [int(uid) for uid in body.get("uids")]
            except:
                logger.error(f"Error in uids: {body.get('uids')}")
        else:
            uids = filter_available_uids(
                task=body.get("task"), model=body.get("model"), test=shared_settings.API_TEST_MODE, n_miners=N_MINERS
            )
        if not uids:
            raise HTTPException(status_code=500, detail="No available miners")

        # Choose between regular completion and mixture of miners.
        if body.get("test_time_inference", False):
            return await test_time_inference(
                body["messages"], body.get("model", None), target_uids=uids
            )
        if body.get("mixture", False):
            return await mixture_of_miners(body, uids=uids)
        else:
            return await chat_completion(body, uids=uids)

    except Exception as e:
        logger.exception(f"Error in chat completion: {e}")
        return StreamingResponse(content="Internal Server Error", status_code=500)


@router.post("/web_retrieval")
async def web_retrieval(
    search_query: str,
    n_miners: int = 10,
    n_results: int = 5,
    max_response_time: int = 10,
    api_key: str = Depends(validate_api_key),
    target_uids: list[str] = None,
):
    if target_uids:
        uids = target_uids
        try:
            uids = [int(uid) for uid in uids]
        except:
            pass
    else:
        uids = filter_available_uids(task="WebRetrievalTask", test=shared_settings.API_TEST_MODE, n_miners=n_miners)
        uids = random.sample(uids, min(len(uids), n_miners))

    if len(uids) == 0:
        raise HTTPException(status_code=500, detail="No available miners")

    body = {
        "seed": random.randint(0, 1_000_000),
        "sampling_parameters": shared_settings.SAMPLING_PARAMS,
        "task": "WebRetrievalTask",
        "target_results": n_results,
        "timeout": max_response_time,
        "messages": [
            {"role": "user", "content": search_query},
        ],
    }

    timeout_seconds = 30
    stream_results = await query_miners(uids, body, timeout_seconds)
    results = [
        "".join(res.accumulated_chunks)
        for res in stream_results
        if isinstance(res, SynapseStreamResult) and res.accumulated_chunks
    ]
    distinct_results = list(np.unique(results))
    loaded_results = []
    for result in distinct_results:
        try:
            loaded_results.append(json.loads(result))
            logger.info(f"🔍 Result: {result}")
        except Exception:
            logger.error(f"🔍 Result: {result}")
    if len(loaded_results) == 0:
        raise HTTPException(status_code=500, detail="No miner responded successfully")

    collected_chunks_list = [res.accumulated_chunks if res and res.accumulated_chunks else [] for res in stream_results]
    asyncio.create_task(scoring_queue.scoring_queue.append_response(uids=uids, body=body, chunks=collected_chunks_list))
    return loaded_results


@router.post("/test_time_inference")
async def test_time_inference(messages: list[dict], model: str = None, target_uids: list[str] = None):
    async def create_response_stream(messages):
        async for steps, total_thinking_time in generate_response(messages, model=model, target_uids=target_uids):
            if total_thinking_time is not None:
                logger.debug(f"**Total thinking time: {total_thinking_time:.2f} seconds**")
            yield steps, total_thinking_time

    # Create a streaming response that yields each step
    async def stream_steps():
        try:
            i = 0
            async for steps, thinking_time in create_response_stream(messages):
                i += 1
                yield "data: " + ChatCompletionChunk(
                    id=str(uuid.uuid4()),
                    created=int(time.time()),
                    model=model or "None",
                    object="chat.completion.chunk",
                    choices=[
                        Choice(index=i, delta=ChoiceDelta(content=f"## {steps[-1][0]}\n\n{steps[-1][1]}" + "\n\n"))
                    ],
                ).model_dump_json() + "\n\n"
        except Exception as e:
            logger.exception(f"Error during streaming: {e}")
            yield f'data: {{"error": "Internal Server Error: {str(e)}"}}\n\n'
        finally:
            yield "data: [DONE]\n\n"

    return StreamingResponse(
        stream_steps(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
        },
    )
