import asyncio
import json
import math
import random
from typing import AsyncGenerator, List, Optional

from fastapi import HTTPException
from fastapi.responses import StreamingResponse
from loguru import logger

from shared.epistula import make_openai_query
from shared.settings import shared_settings
from shared.uids import get_uids
from validator_api.utils import forward_response


async def stream_from_first_response(
    responses: List[asyncio.Task], collected_chunks_list: List[List[str]], body: dict[str, any], uids: List[int]
) -> AsyncGenerator[str, None]:
    first_valid_response = None
    try:
        # Wait for the first valid response
        while responses and first_valid_response is None:
            done, pending = await asyncio.wait(responses, return_when=asyncio.FIRST_COMPLETED)

            for task in done:
                try:
                    response = await task
                    if response and not isinstance(response, Exception):
                        first_valid_response = response
                        break
                except Exception as e:
                    logger.error(f"Error in miner response: {e}")
                responses.remove(task)

        if first_valid_response is None:
            logger.error("No valid response received from any miner")
            yield 'data: {"error": "502 - No valid response received"}\n\n'
            return

        # Stream the first valid response
        chunks_received = False
        async for chunk in first_valid_response:
            chunks_received = True
            collected_chunks_list[0].append(chunk.choices[0].delta.content)
            yield f"data: {json.dumps(chunk.model_dump())}\n\n"

        if not chunks_received:
            logger.error("Stream is empty: No chunks were received")
            yield 'data: {"error": "502 - Response is empty"}\n\n'

        yield "data: [DONE]\n\n"

        # Continue collecting remaining responses in background for scoring
        remaining = asyncio.gather(*pending, return_exceptions=True)
        asyncio.create_task(collect_remaining_responses(remaining, collected_chunks_list, body, uids))

    except asyncio.CancelledError:
        logger.info("Client disconnected, streaming cancelled")
        for task in responses:
            task.cancel()
        raise
    except Exception as e:
        logger.exception(f"Error during streaming: {e}")
        yield 'data: {"error": "Internal server Error"}\n\n'


async def collect_remaining_responses(
    remaining: asyncio.Task, collected_chunks_list: List[List[str]], body: dict[str, any], uids: List[int]
):
    """Collect remaining responses for scoring without blocking the main response."""
    try:
        responses = await remaining
        logger.debug(f"responses to forward: {responses}")
        for i, response in enumerate(responses):
            if isinstance(response, Exception):
                logger.error(f"Error collecting response from uid {uids[i+1]}: {response}")
                continue

            async for chunk in response:
                collected_chunks_list[i + 1].append(chunk.choices[0].delta.content)
        for uid, chunks in zip(uids, collected_chunks_list):
            # Forward for scoring
            asyncio.create_task(forward_response(uid, body, chunks))

    except Exception as e:
        logger.exception(f"Error collecting remaining responses: {e}")


async def get_response_from_miner(body: dict[str, any], uid: int) -> tuple:
    """Get response from a single miner."""
    return await make_openai_query(shared_settings.METAGRAPH, shared_settings.WALLET, body, uid, stream=False)


async def chat_completion(
    body: dict[str, any], uids: Optional[list[int]] = None, num_miners: int = 3
) -> tuple | StreamingResponse:
    """Handle chat completion with multiple miners in parallel."""
    # Get multiple UIDs if none specified
    if uids is None:
        uids = list(get_uids(sampling_mode="top_incentive", k=100))
        if uids is None or len(uids) == 0:  # if not uids throws error, figure out how to fix
            logger.error("No available miners found")
            raise HTTPException(status_code=503, detail="No available miners found")
        selected_uids = random.sample(uids, min(num_miners, len(uids)))
    else:
        selected_uids = uids[:num_miners]  # If UID is specified, only use that one

    logger.debug(f"Querying uids {selected_uids}")
    STREAM = body.get("stream", False)

    # Initialize chunks collection for each miner
    collected_chunks_list = [[] for _ in selected_uids]

    timeout_seconds = int(math.floor(math.log2(body["sampling_parameters"]["max_new_tokens"] / 256))) * 10 + 30
    if STREAM:
        # Create tasks for all miners
        response_tasks = [
            asyncio.create_task(
                make_openai_query(
                    shared_settings.METAGRAPH, shared_settings.WALLET, timeout_seconds, body, uid, stream=True
                )
            )
            for uid in selected_uids
        ]

        return StreamingResponse(
            stream_from_first_response(response_tasks, collected_chunks_list, body, selected_uids),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
            },
        )
    else:
        # For non-streaming requests, wait for first valid response
        response_tasks = [asyncio.create_task(get_response_from_miner(body, uid)) for uid in selected_uids]

        first_valid_response = None
        collected_responses = []

        while response_tasks and first_valid_response is None:
            done, pending = await asyncio.wait(response_tasks, return_when=asyncio.FIRST_COMPLETED)

            for task in done:
                try:
                    response = await task
                    if response and isinstance(response, tuple):
                        if first_valid_response is None:
                            first_valid_response = response
                        collected_responses.append(response)
                except Exception as e:
                    logger.error(f"Error in miner response: {e}")
                response_tasks.remove(task)

        if first_valid_response is None:
            raise HTTPException(status_code=502, detail="No valid response received")

        # Forward all collected responses for scoring in the background
        for i, response in enumerate(collected_responses):
            if response and isinstance(response, tuple):
                asyncio.create_task(forward_response(uid=selected_uids[i], body=body, chunks=response[1]))

        return first_valid_response[0]  # Return only the response object, not the chunks
