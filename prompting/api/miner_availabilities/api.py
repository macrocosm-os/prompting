from typing import Literal
import json

from fastapi import APIRouter
from fastapi.encoders import jsonable_encoder

from prompting.miner_availability.miner_availability import miner_availabilities
from prompting.tasks.task_registry import TaskRegistry

router = APIRouter()


@router.post("/miner_availabilities")
async def get_miner_availabilities(uids: list[int] | None = None):
    if uids:
        data = {uid: miner_availabilities.miners.get(uid) for uid in uids}
        encoded_data = jsonable_encoder(data)
        with open("miner_availabilities.json", "w") as file:
            json.dump(encoded_data, file)
        return data
    return miner_availabilities.miners


@router.get("/get_available_miners")
async def get_available_miners(
    task: Literal[tuple([config.task.__name__ for config in TaskRegistry.task_configs])] | None = None,
    model: str | None = None,
    k: int = 10,
):
    task_configs = [config for config in TaskRegistry.task_configs if config.task.__name__ == task]
    task_config = task_configs[0] if task_configs else None
    return miner_availabilities.get_available_miners(task=task_config, model=model, k=k)
