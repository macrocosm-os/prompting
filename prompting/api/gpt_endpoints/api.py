import json
import random
from typing import AsyncGenerator

from fastapi import APIRouter, Depends, HTTPException, Request
from loguru import logger
from starlette.responses import StreamingResponse

from prompting.api.api_managements.api import validate_api_key
from prompting.base.dendrite import DendriteResponseEvent, SynapseStreamResult
from prompting.miner_availability.miner_availability import miner_availabilities
from prompting.rewards.scoring import task_scorer
from prompting.settings import settings
from prompting.tasks.inference import InferenceTask
from prompting.tasks.task_registry import TaskRegistry
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
                yield f"data: {json.dumps(chunk.model_dump())}\n\n"

    task = InferenceTask(
        query=request["messages"][-1]["content"],
        messages=[message["content"] for message in request["messages"]],
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
        completions=["".join(collected_content)],
    )

    task_scorer.add_to_queue(
        task=task, response=response_event, dataset_entry=task.dataset_entry, block=-1, step=-1, task_id=task.task_id
    )
    yield "data: [DONE]\n\n"


@router.post("/mixture_of_agents")
async def mixture_of_agents(request: Request, api_key_data: dict = Depends(validate_api_key)):
    return {"message": "Mixture of Agents"}


import asyncio

from prompting.base.epistula import make_openai_query


async def query_endpoint(metagraph, wallet, body, uid, stream):
    try:
        response = await make_openai_query(metagraph=metagraph, wallet=wallet, body=body, uid=uid, stream=stream)
        if stream:

            async def stream_response():
                collected_content = []
                collected_chunks_timings = []
                with Timer() as timer:
                    async for chunk in response:
                        if (
                            hasattr(chunk, "choices")
                            and chunk.choices
                            and isinstance(chunk.choices[0].delta.content, str)
                        ):
                            content = chunk.choices[0].delta.content
                            collected_content.append(content)
                            collected_chunks_timings.append(timer.elapsed_time())
                            yield f"data: {json.dumps(chunk.model_dump())}\n\n"

                    # Add task to scoring queue after stream completes
                    task_obj = InferenceTask(
                        query=body["messages"][-1]["content"],
                        messages=[message["content"] for message in body["messages"]],
                        model=body.get("model"),
                        seed=body.get("seed"),
                        response="".join(collected_content),
                    )
                    response_event = DendriteResponseEvent(
                        stream_results=[
                            SynapseStreamResult(
                                uid=uid,
                                accumulated_chunks=collected_content,
                                accumulated_chunks_timings=collected_chunks_timings,
                            )
                        ],
                        uids=[uid],
                        timeout=settings.NEURON_TIMEOUT,
                        completions=["".join(collected_content)],
                    )
                    task_scorer.add_to_queue(
                        task=task_obj,
                        response=response_event,
                        dataset_entry=task_obj.dataset_entry,
                        block=-1,
                        step=-1,
                        task_id=task_obj.task_id,
                    )
                    yield "data: [DONE]\n\n"

            return StreamingResponse(stream_response(), media_type="text/event-stream")
        else:
            return response

    except Exception as e:
        logger.error(f"Error querying miner with uid {uid}: {e}")
        return None


async def query_all_endpoints(metagraph, wallet, body, uids, stream):
    tasks = [query_endpoint(metagraph, wallet, body, uid, stream) for uid in uids]
    for task in asyncio.as_completed(tasks):
        result = await task
        if result is not None:
            return result
    raise HTTPException(status_code=503, detail="No valid response from any endpoint")


# TODO: Modify this so ALL of the responses are added for scoring rather than just the first one
@router.post("/v1/chat/completions")
async def proxy_chat_completions(request: Request, api_key_data: dict = Depends(validate_api_key)):
    body = await request.json()
    task = TaskRegistry.get_task_by_name(body.get("task"))
    if body.get("task") and not task:
        raise HTTPException(status_code=400, detail=f"Task {body.get('task')} not found")
    logger.debug(f"Requested Task: {body.get('task')}, {task}")

    stream = body.get("stream", False)
    body = {k: v for k, v in body.items() if k not in ["task", "stream"]}
    body["task"] = task.__class__.__name__
    body["seed"] = body.get("seed") or str(random.randint(0, 1_000_000))
    logger.debug(f"Seed provided by miner: {bool(body.get('seed'))} -- Using seed: {body.get('seed')}")

    # Get available miners
    if uids := body.get("uids"):
        available_miners = uids
    elif settings.TEST_MINER_IDS:
        available_miners = settings.TEST_MINER_IDS
    elif not settings.mode == "mock" and not (
        available_miners := miner_availabilities.get_available_miners(task=task, model=body.get("model"))
    ):
        raise HTTPException(
            status_code=503,
            detail=f"No miners available for model: {body.get('model')} and task: {task.__class__.__name__}",
        )
    random.shuffle(available_miners)

    try:
        result = await query_all_endpoints(settings.METAGRAPH, settings.WALLET, body, available_miners, stream)
        if not stream:
            # Handle non-streaming response scoring
            response_event = DendriteResponseEvent(
                stream_results=[
                    SynapseStreamResult(
                        uid=available_miners[0],
                        accumulated_chunks=[result.choices[0].message.content],
                        accumulated_chunks_timings=[0.0],
                    )
                ],
                uids=[available_miners[0]],
                timeout=settings.NEURON_TIMEOUT,
                completions=[result.choices[0].message.content],
            )
            task_obj = InferenceTask(
                query=body["messages"][-1]["content"],
                messages=[message["content"] for message in body["messages"]],
                model=body.get("model"),
                seed=body.get("seed"),
                response=result.choices[0].message.content,
            )
            task_scorer.add_to_queue(
                task=task_obj,
                response=response_event,
                dataset_entry=task_obj.dataset_entry,
                block=-1,
                step=-1,
                task_id=task_obj.task_id,
            )
        return result
    except Exception as e:
        logger.error(f"Failed to get a valid response: {e}")
        raise HTTPException(status_code=503, detail=str(e))
