from fastapi import APIRouter, Request
from loguru import logger
import random
import openai
from prompting.settings import settings
from httpx import Timeout
from prompting.base.epistula import create_header_hook
from fastapi.responses import StreamingResponse
import json
from prompting.miner_availability.miner_availability import miner_availabilities
from prompting.tasks.inference import InferenceTask
from typing import AsyncGenerator
from prompting.rewards.scoring import task_scorer
from prompting.base.dendrite import DendriteResponseEvent, SynapseStreamResult
from prompting.utils.timer import Timer

router = APIRouter()


async def process_and_collect_stream(miner_id: int, request: dict, response: AsyncGenerator):
    collected_content = []
    collected_chunks_timings = []
    with Timer() as timer:
        async for chunk in response:
            logger.debug(f"Chunk: {chunk}")
            if hasattr(chunk, "choices") and chunk.choices and isinstance(chunk.choices[0].delta.content, str):
                collected_content.append(chunk.choices[0].delta.content)
                collected_chunks_timings.append(timer.elapsed_time())
                # Format in SSE format
                yield f"data: {json.dumps(chunk.model_dump())}\n\n"
        # After streaming is complete, put the response in the queue
    task = InferenceTask(
        query=request["messages"][-1]["content"],
        model=request.get("model"),
        seed=request.get("seed"),
        response="".join(collected_content),
    )
    logger.debug(f"Adding Organic Request to scoring queue: {task}")
    response_event = DendriteResponseEvent(
        stream_results=[
            SynapseStreamResult(
                uid=miner_id,
                accumulated_chunks=collected_content,
                accumulated_chunks_timings=collected_chunks_timings,
            )
        ],
        uids=[miner_id],
        timeout=settings.NEURON_TIMEOUT,
    )

    # TODO: Estimate block and step
    task_scorer.add_to_queue(
        task=task, response=response_event, dataset_entry=task.dataset_entry, block=-1, step=-1, task_id=task.task_id
    )
    yield "data: [DONE]\n\n"


@router.post("/v1/chat/completions")
async def proxy_chat_completions(request: Request):
    # Get the request body
    body = await request.json()
    body["seed"] = body.get("seed") or str(
        random.randint(0, 1_000_000)
    )  # for some reason needs to be passed as string... it seems?

    # Ensure streaming is enabled
    # body["stream"] = True
    if settings.TEST_MINER_IDS:
        available_miners = settings.TEST_MINER_IDS
    elif not settings.mode == "mock" and not (
        available_miners := miner_availabilities.get_available_miners(task=InferenceTask(), model=None)
    ):
        return "No miners available"
    axon_info = settings.METAGRAPH.axons[available_miners[0]]

    # TODO: Remove this/build better testing mechanism
    base_url = "http://localhost:8008/v1" if settings.mode == "mock" else f"http://{axon_info.ip}:{axon_info.port}/v1"
    # base_url = "http://localhost:8008/v1"
    # available_miners = [-1]

    miner_id = available_miners[0]

    logger.debug(f"Using base_url: {base_url}")

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
    # return await process_and_collect_stream(process_stream(response))
    return StreamingResponse(process_and_collect_stream(miner_id, body, response), media_type="text/event-stream")
