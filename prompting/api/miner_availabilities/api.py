from fastapi import APIRouter
from prompting.miner_availability.miner_availability import miner_availabilities
from loguru import logger

router = APIRouter()


@router.post("/miner_availabilities")
async def get_miner_availabilities(uids: list[int] | None = None):
    if uids:
        return {uid: miner_availabilities.miners.get(uid) for uid in uids}
    logger.info(f"Returning all miner availabilities for {len(miner_availabilities.miners)} miners")
    return miner_availabilities.miners


@router.get("/get_available_miners")
async def get_available_miners(task: str | None = None, model: str | None = None, k: int = 10):
    logger.info(f"Getting {k} available miners for task {task} and model {model}")
    return miner_availabilities.get_available_miners(task=task, model=model, k=k)
