import json
import random
from typing import AsyncGenerator

from fastapi import APIRouter, Depends, HTTPException, Request
from loguru import logger

from prompting.api.api_managements.api import validate_api_key
from prompting.base.dendrite import DendriteResponseEvent, SynapseStreamResult
from prompting.base.epistula import query_miners
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


@router.post("/v1/chat/completions")
async def proxy_chat_completions(request: Request, api_key_data: dict = Depends(validate_api_key)):
    body = await request.json()
    task = TaskRegistry.get_task_by_name(body.get("task"))
    if body.get("task") and not task:
        raise HTTPException(status_code=400, detail=f"Task {body.get('task')} not found")
    logger.debug(f"Requested Task: {body.get('task')}, {task}")

    stream = body.get("stream")
    body = {k: v for k, v in body.items() if k not in ["task", "stream"]}
    body["task"] = task.__class__.__name__
    body["seed"] = body.get("seed") or str(random.randint(0, 1_000_000))
    logger.debug(f"Seed provided by miner: {bool(body.get('seed'))} -- Using seed: {body.get('seed')}")

    if settings.TEST_MINER_IDS:
        available_miners = settings.TEST_MINER_IDS
    elif not settings.mode == "mock" and not (
        available_miners := miner_availabilities.get_available_miners(task=task, model=body.get("model"))
    ):
        raise HTTPException(
            status_code=503,
            detail=f"No miners available for model: {body.get('model')} and task: {task.__class__.__name__}",
        )

    response = query_miners(available_miners, json.dumps(body).encode("utf-8"), stream=stream)
    if stream:
        return response
    else:
        response = await response
        response_event = DendriteResponseEvent(
            stream_results=response,
            uids=available_miners,
            timeout=settings.NEURON_TIMEOUT,
            completions=["".join(res.accumulated_chunks) for res in response],
        )

        task = task(
            query=body["messages"][-1]["content"],
            messages=[message["content"] for message in body["messages"]],
            model=body.get("model"),
            seed=body.get("seed"),
            response=response_event,
        )

        task_scorer.add_to_queue(
            task=task,
            response=response_event,
            dataset_entry=task.dataset_entry,
            block=-1,
            step=-1,
            task_id=task.task_id,
        )

        return [res.model_dump() for res in response]

    # axon_info = settings.METAGRAPH.axons[available_miners[0]]
    # base_url = "http://localhost:8008/v1" if settings.mode == "mock" else f"http://{axon_info.ip}:{axon_info.port}/v1"
    # miner_id = available_miners[0]
    # logger.debug(f"Using base_url: {base_url}")

    # miner = openai.AsyncOpenAI(
    #     base_url=base_url,
    #     max_retries=0,
    #     timeout=Timeout(settings.NEURON_TIMEOUT, connect=5, read=5),
    #     http_client=openai.DefaultAsyncHttpxClient(
    #         event_hooks={"request": [create_header_hook(settings.WALLET.hotkey, axon_info.hotkey)]}
    #     ),
    # )

    # try:
    #     with Timer() as timer:
    #         if task:
    #             response = await miner.chat.completions.create(**body, extra_body={"task": task.__name__})
    #         else:
    #             response = await miner.chat.completions.create(**body)
    #     if body.get("stream"):
    #         return StreamingResponse(
    #             process_and_collect_stream(miner_id, body, response), media_type="text/event-stream"
    #         )
    # except Exception as e:
    #     logger.exception(f"Error coming from Miner: {e}")
    #     raise HTTPException(status_code=500, detail=f"Error coming from Miner: {e}")

    # response_event = DendriteResponseEvent(
    #     stream_results=[
    #         SynapseStreamResult(
    #             uid=miner_id,
    #             accumulated_chunks=[response.choices[0].message.content],
    #             accumulated_chunks_timings=[timer.final_time],
    #         )
    #     ],
    #     completions=[response.choices[0].message.content],
    #     uids=[miner_id],
    #     timeout=settings.NEURON_TIMEOUT,
    # )

    # task = InferenceTask(
    #     query=body["messages"][-1]["content"],
    #     messages=[message["content"] for message in body["messages"]],
    #     model=body.get("model"),
    #     seed=body.get("seed"),
    #     response=response_event,
    # )

    # task_scorer.add_to_queue(
    #     task=task, response=response_event, dataset_entry=task.dataset_entry, block=-1, step=-1, task_id=task.task_id
    # )

    # return response
