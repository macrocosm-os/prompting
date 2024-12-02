import json
import random

from fastapi import HTTPException
from loguru import logger

from prompting.base.dendrite import DendriteResponseEvent
from prompting.base.epistula import query_miners
from prompting.miner_availability.miner_availability import miner_availabilities
from prompting.rewards.scoring import task_scorer
from prompting.settings import settings
from prompting.tasks.inference import InferenceTask
from prompting.tasks.task_registry import TaskRegistry


async def process_completions(body: dict[str, any]):
    task = TaskRegistry.get_task_by_name(body.get("task", InferenceTask.__name__))
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
            detail=f"No miners available for model: {body.get('model')} and task: {task.__name__}",
        )

    response = query_miners(available_miners, json.dumps(body).encode("utf-8"), stream=stream)
    if stream:
        return response

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
