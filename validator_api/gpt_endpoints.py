import asyncio
import json
import random
import time
import uuid

import numpy as np
from fastapi import APIRouter, Depends, HTTPException, status
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
from validator_api.serializers import (
    CompletionsRequest,
    InferenceRequest,
    WebRetrievalRequest,
    WebRetrievalResponse,
    WebSearchResult,
)
from validator_api.test_time_inference import generate_response
from validator_api.utils import filter_available_uids

router = APIRouter()
N_MINERS = 5


@router.post(
    "/v1/chat/completions",
    summary="Chat completions endpoint",
    description="Main endpoint that handles both regular, multi step reasoning, test time inference, and mixture of miners chat completion.",
    response_description="Streaming response with generated text",
    status_code=status.HTTP_200_OK,
    responses={
        status.HTTP_200_OK: {
            "description": "Successful response with streaming text",
            "content": {"text/event-stream": {}},
        },
        status.HTTP_500_INTERNAL_SERVER_ERROR: {"description": "Internal server error or no available miners"},
    },
)
async def completions(request: CompletionsRequest, api_key: str = Depends(validate_api_key)):
    """
    Chat completions endpoint that supports different inference modes.

    This endpoint processes chat messages and returns generated completions using
    different inference strategies based on the request parameters.

    ## Inference Modes:
    - Regular chat completion
    - Multi Step Reasoning
    - Test time inference
    - Mixture of miners

    ## Request Parameters:
    - **uids** (List[int], optional): Specific miner UIDs to query. If not provided, miners will be selected automatically.
    - **messages** (List[dict]): List of message objects with 'role' and 'content' keys. Required.
    - **seed** (int, optional): Random seed for reproducible results.
    - **task** (str, optional): Task identifier to filter available miners.
    - **model** (str, optional): Model identifier to filter available miners.
    - **test_time_inference** (bool, default=False): Enable step-by-step reasoning mode.
    - **mixture** (bool, default=False): Enable mixture of miners mode.
    - **sampling_parameters** (dict, optional): Parameters to control text generation.

    The endpoint selects miners based on the provided UIDs or filters available miners
    based on task and model requirements.

    Example request:
    ```json
    {
      "messages": [
        {"role": "user", "content": "Tell me about neural networks"}
      ],
      "model": "gpt-4",
      "seed": 42
    }
    ```
    """
    try:
        body = request.model_dump()
        if body.get("inference_mode") == "Reasoning-Fast":
            body["task"] = "MultiStepReasoningTask"
        if body.get("model") == "Default":
            # By setting default, we are allowing a user to use whatever model we define as the standard, could also set to None.
            body["model"] = "hugging-quants/Meta-Llama-3.1-70B-Instruct-AWQ-INT4"
        body["seed"] = int(body.get("seed") or random.randint(0, 1000000))
        if body.get("uids"):
            try:
                uids = list(map(int, body.get("uids")))
            except Exception:
                logger.error(f"Error in uids: {body.get('uids')}")
        else:
            uids = filter_available_uids(
                task=body.get("task"), model=body.get("model"), test=shared_settings.API_TEST_MODE, n_miners=N_MINERS
            )
        if not uids:
            raise HTTPException(status_code=500, detail="No available miners")

        if body.get("test_time_inference", False) or body.get("inference_mode", None) == "Chain-of-Thought":
            test_time_request = InferenceRequest(
                messages=request.messages,
                model=request.model,
                uids=uids if uids else None,
                json_format=request.json_format,
            )
            return await test_time_inference(test_time_request)
        elif body.get("mixture", False) or body.get("inference_mode", None) == "Mixture-of-Agents":
            return await mixture_of_miners(body, uids=uids)
        else:
            return await chat_completion(body, uids=uids)

    except Exception as e:
        logger.exception(f"Error in chat completion: {e}")
        return StreamingResponse(content="Internal Server Error", status_code=500)


@router.post(
    "/web_retrieval",
    response_model=WebRetrievalResponse,
    summary="Web retrieval endpoint",
    description="Retrieves information from the web based on a search query using multiple miners.",
    response_description="List of unique web search results",
    status_code=status.HTTP_200_OK,
    responses={
        status.HTTP_200_OK: {
            "description": "Successful response with web search results",
            "model": WebRetrievalResponse,
        },
        status.HTTP_500_INTERNAL_SERVER_ERROR: {
            "description": "Internal server error, no available miners, or no successful miner responses"
        },
    },
)
async def web_retrieval(
    request: WebRetrievalRequest,
    api_key: str = Depends(validate_api_key),
):
    """
    Web retrieval endpoint that queries multiple miners to search the web.

    This endpoint distributes a search query to multiple miners, which perform web searches
    and return relevant results. The results are deduplicated based on URLs before being returned.

    ## Request Parameters:
    - **search_query** (str): The query to search for on the web. Required.
    - **n_miners** (int, default=10): Number of miners to query for results.
    - **n_results** (int, default=5): Maximum number of results to return in the response.
    - **max_response_time** (int, default=10): Maximum time to wait for responses in seconds.
    - **uids** (List[int], optional): Optional list of specific miner UIDs to query.

    ## Response:
    Returns a list of unique web search results, each containing:
    - **url** (str): The URL of the web page
    - **content** (str, optional): The relevant content from the page
    - **relevant** (str, optional): Information about why this result is relevant

    Example request:
    ```json
    {
      "search_query": "latest advancements in quantum computing",
      "n_miners": 15,
      "n_results": 10
    }
    ```
    """
    if request.uids:
        uids = request.uids
        try:
            uids = list(map(int, uids))
        except Exception:
            logger.error(f"Error in uids: {uids}")
    else:
        uids = filter_available_uids(
            task="WebRetrievalTask", test=shared_settings.API_TEST_MODE, n_miners=request.n_miners
        )
        uids = random.sample(uids, min(len(uids), request.n_miners))

    if len(uids) == 0:
        raise HTTPException(status_code=500, detail="No available miners")

    body = {
        "seed": random.randint(0, 1_000_000),
        "sampling_parameters": shared_settings.SAMPLING_PARAMS,
        "task": "WebRetrievalTask",
        "target_results": request.n_results,
        "timeout": request.max_response_time,
        "messages": [
            {"role": "user", "content": request.search_query},
        ],
    }

    timeout_seconds = 30  # TODO: We need to scale down this timeout
    logger.debug(f"🔍 Querying miners: {uids} for web retrieval")
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
    loaded_results = [json.loads(r) if isinstance(r, str) else r for r in loaded_results]
    flat_results = [item for sublist in loaded_results for item in sublist]
    unique_results = []
    seen_urls = set()

    for result in flat_results:
        if isinstance(result, dict) and "url" in result:
            if result["url"] not in seen_urls:
                seen_urls.add(result["url"])
                # Convert dict to WebSearchResult
                unique_results.append(WebSearchResult(**result))

    return WebRetrievalResponse(results=unique_results)


@router.post("/test_time_inference")
async def test_time_inference(request: InferenceRequest):
    """
    Test time inference endpoint that provides step-by-step reasoning.

    This endpoint streams the thinking process and reasoning steps during inference,
    allowing visibility into how the model arrives at its conclusions. Each step of
    the reasoning process is streamed as it becomes available.

    ## Request Parameters:
    - **messages** (List[dict]): List of message objects with 'role' and 'content' keys. Required.
    - **model** (str, optional): Optional model identifier to use for inference.
    - **uids** (List[int], optional): Optional list of specific miner UIDs to query.

    ## Response:
    The response is streamed as server-sent events (SSE) with each step of reasoning.
    Each event contains:
    - A step title/heading
    - The content of the reasoning step
    - Timing information (debug only)

    Example request:
    ```json
    {
      "messages": [
        {"role": "user", "content": "Solve the equation: 3x + 5 = 14"}
      ],
      "model": "gpt-4"
    }
    ```
    """

    async def create_response_stream(request):
        async for steps, total_thinking_time in generate_response(
            request.messages, model=request.model, uids=request.uids
        ):
            if total_thinking_time is not None:
                logger.debug(f"**Total thinking time: {total_thinking_time:.2f} seconds**")
            yield steps, total_thinking_time

    # Create a streaming response that yields each step
    async def stream_steps():
        try:
            i = 0
            async for steps, thinking_time in create_response_stream(request):
                i += 1
                if request.json_format:
                    choice = Choice(index=i, delta=ChoiceDelta(content=json.dumps(steps[-1])))
                else:
                    choice = Choice(index=i, delta=ChoiceDelta(content=f"## {steps[-1][0]}\n\n{steps[-1][1]}" + "\n\n"))
                yield "data: " + ChatCompletionChunk(
                    id=str(uuid.uuid4()),
                    created=int(time.time()),
                    model=request.model or "None",
                    object="chat.completion.chunk",
                    choices=[choice],
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
