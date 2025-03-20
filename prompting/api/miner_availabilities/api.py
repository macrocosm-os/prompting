from typing import Literal

from fastapi import APIRouter, Request

from prompting.miner_availability.miner_availability import MinerAvailabilities
from prompting.tasks.task_registry import TaskRegistry

router = APIRouter()


@router.post("/miner_availabilities")
async def get_miner_availabilities(request: Request, uids: list[int] | None = None):
    if uids:
        return {uid: request.app.state.miners_dict.get(uid) for uid in uids}
    return request.app.state.miners_dict


@router.get("/get_available_miners")
async def get_available_miners(
    request: Request,
    task: Literal[tuple([config.task.__name__ for config in TaskRegistry.task_configs])] | None = None,
    model: str | None = None,
    k: int = 10,
):
    task_configs = [config for config in TaskRegistry.task_configs if config.task.__name__ == task]
    task_config = task_configs[0] if task_configs else None
    return MinerAvailabilities.get_available_miners(
        miners=request.app.state.miners_dict, task=task_config, model=model, k=k
    )
