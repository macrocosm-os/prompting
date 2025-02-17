import asyncio
import json
import random
import time
import uuid

import numpy as np
from fastapi import APIRouter, Depends, Header, HTTPException
from loguru import logger
from openai.types.chat.chat_completion_chunk import ChatCompletionChunk, Choice, ChoiceDelta
from starlette.responses import StreamingResponse

from shared import settings
from shared.epistula import SynapseStreamResult, query_miners
from validator_api import scoring_queue
from validator_api.api_management import _keys
from validator_api.chat_completion import chat_completion
from validator_api.mixture_of_miners import mixture_of_miners
from validator_api.test_time_inference import generate_response
from validator_api.utils import filter_available_uids

from .serializers import ChatCompletionRequest, ErrorResponse, SearchResult, WebSearchResponse

shared_settings = settings.shared_settings

router = APIRouter()
N_MINERS = 5


def validate_api_key(api_key: str = Header(...)):
    """Validates API key for authentication."""
    if api_key not in _keys:
        raise HTTPException(status_code=403, detail="Invalid API key")
    return _keys[api_key]


@router.post(
    "/v1/chat/completions",
    response_model=WebSearchResponse,
    responses={
        200: {
            "description": "Successfully generated chat completion",
            "model": WebSearchResponse,
            "content": {"application/json": {"example": {"results": [{"content": "This is a sample response..."}]}}},
        },
        403: {
            "description": "Invalid API key provided",
            "model": ErrorResponse,
            "content": {"application/json": {"example": {"detail": "Invalid API key"}}},
        },
        500: {
            "description": "Server error occurred",
            "model": ErrorResponse,
            "content": {"application/json": {"example": {"detail": "No available miners"}}},
        },
    },
    summary="Generate chat completions",
    description="""
    Generates chat completions using various strategies:

    - Standard chat completion
    - Mixture-of-miners strategy
    - Test-time inference

    The endpoint automatically selects the appropriate strategy based on request parameters.
    Results are streamed back to the client as they are generated.
    """,
)
async def completions(request: ChatCompletionRequest, api_key: str = Depends(validate_api_key)):
    """
    Executes a chat completion request.

    - **request**: JSON request body following `ChatCompletionRequest` model.
    - **api_key**: Authentication header for API access.

    Determines whether to use:
    - Standard chat completion (`chat_completion`).
    - Mixture-of-miners strategy (`mixture_of_miners`).
    - Test-time inference (`test_time_inference`).
    """
    try:
        body = await request.json()
        body["seed"] = int(body.get("seed") or random.randint(0, 1000000))
        uids = body.get("uids") or filter_available_uids(
            task=body.get("task"), model=body.get("model"), test=shared_settings.API_TEST_MODE, n_miners=N_MINERS
        )
        if not uids:
            raise HTTPException(status_code=500, detail="No available miners")

        if request.test_time_inference:
            return await test_time_inference(request.messages, request.model)

        if request.mixture:
            return await mixture_of_miners(request.dict(), uids=uids)

        return await chat_completion(request.dict(), uids=uids)

    except Exception as e:
        logger.exception(f"Error in chat completion: {e}")
        return StreamingResponse(content="Internal Server Error", status_code=500)


@router.post(
    "/web_retrieval",
    response_model=WebSearchResponse,
    responses={
        200: {"description": "Successfully retrieved search results", "model": WebSearchResponse},
        403: {"description": "Invalid API key provided", "model": ErrorResponse},
        422: {"description": "Invalid request parameters", "model": ErrorResponse},
        500: {"description": "Server error occurred", "model": ErrorResponse},
    },
    summary="Search the web using distributed miners",
    description="""
    Executes web searches using a distributed network of miners:

    1. Queries multiple miners in parallel
    2. Aggregates and deduplicates results
    3. Parses and validates all responses
    4. Returns a unified set of search results

    The search is performed using DuckDuckGo through the miner network.
    """,
)
async def web_retrieval(search_query: str, n_miners: int = 10, n_results: int = 5, max_response_time: int = 10):
    """
    Handles web retrieval through distributed miners.

    - **request**: JSON request body following `WebSearchQuery` model.

    If no miners are available, an HTTPException is raised.
    """
    uids = filter_available_uids(task="WebRetrievalTask", test=shared_settings.API_TEST_MODE, n_miners=n_miners)
    if not uids:
        raise HTTPException(status_code=500, detail="No available miners")

    uids = random.sample(uids, min(len(uids), n_miners))
    logger.debug(f"🔍 Querying uids: {uids}")

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
    logger.info(f"🔍 {len(results)} miners responded successfully with {len(distinct_results)} distinct results.")

    search_results = []
    for result in distinct_results:
        try:
            parsed_result = json.loads(result)
            search_results.append(SearchResult(**parsed_result))
            logger.info(f"🔍 Parsed Result: {parsed_result}")
        except Exception:
            logger.error(f"🔍 Failed to parse result: {result}")

    if len(search_results) == 0:
        raise HTTPException(status_code=500, detail="No miner responded successfully")

    asyncio.create_task(scoring_queue.scoring_queue.append_response(uids=uids, body=body, chunks=[]))
    return WebSearchResponse(results=search_results)


@router.post(
    "/test_time_inference",
    responses={
        200: {"description": "Successfully generated inference response", "content": {"text/event-stream": {}}},
        500: {"description": "Server error occurred", "model": ErrorResponse},
    },
    summary="Generate responses using test-time inference",
    description="""
    Generates responses using test-time inference strategy:

    - Streams response steps as they are generated
    - Includes thinking time metrics
    - Returns results in a streaming event format
    """,
)
async def test_time_inference(messages: list[dict], model: str | None = None):
    """
    Handles test-time inference requests.

    - **messages**: List of messages used for inference.
    - **model**: Optional model to use for generating responses.

    Returns a streaming response of the generated chat output.
    """

    async def create_response_stream(messages):
        async for steps, total_thinking_time in generate_response(messages, model=model):
            if total_thinking_time is not None:
                logger.info(f"**Total thinking time: {total_thinking_time:.2f} seconds**")
            yield steps, total_thinking_time

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
