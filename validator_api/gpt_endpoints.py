import asyncio
import json
import random

import numpy as np
from fastapi import APIRouter, Depends, Header, HTTPException, Request
from loguru import logger
from starlette.responses import StreamingResponse

from shared.epistula import SynapseStreamResult, query_miners
from shared.settings import shared_settings
from shared.uids import get_uids
from validator_api.chat_completion import chat_completion
from validator_api.mixture_of_miners import mixture_of_miners
from validator_api.utils import forward_response

router = APIRouter()

# load api keys from api_keys.json
with open("api_keys.json", "r") as f:
    _keys = json.load(f)


def validate_api_key(api_key: str = Header(...)):
    if api_key not in _keys:
        raise HTTPException(status_code=403, detail="Invalid API key")
    return _keys[api_key]


@router.post("/v1/chat/completions")
async def completions(request: Request, api_key: str = Depends(validate_api_key)):
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


@router.post("/web_retrieval")
async def web_retrieval(search_query: str, n_miners: int = 10, uids: list[int] = None):
    uids = list(get_uids(sampling_mode="random", k=n_miners))
    logger.debug(f"🔍 Querying uids: {uids}")
    if len(uids) == 0:
        logger.warning("No available miners. This should already have been caught earlier.")
        return

    body = {
        "seed": random.randint(0, 1_000_000),
        "sampling_parameters": shared_settings.SAMPLING_PARAMS,
        "task": "WebRetrievalTask",
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
    logger.info(
        f"🔍 Collected responses from {len(stream_results)} miners. {len(results)} responded successfully with a total of {len(distinct_results)} distinct results"
    )
    loaded_results = []
    for result in distinct_results:
        try:
            loaded_results.append(json.loads(result))
            logger.info(f"🔍 Result: {result}")
        except Exception:
            logger.error(f"🔍 Result: {result}")
    if len(loaded_results) == 0:
        raise HTTPException(status_code=500, detail="No miner responded successfully")

    for uid, res in zip(uids, stream_results):
        asyncio.create_task(
            forward_response(
                uid=uid, body=body, chunks=res.accumulated_chunks if res and res.accumulated_chunks else []
            )
        )
    return loaded_results
