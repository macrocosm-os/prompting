import asyncio
from collections import deque
import datetime
import httpx
import requests
from loguru import logger

from shared.loop_runner import AsyncLoopRunner
from shared.settings import shared_settings
from shared.uids import get_uids


_scoring_lock = asyncio.Lock()
_scoring_last_query_time = datetime.datetime.fromtimestamp(0)
_scoring_queue: deque[dict[str, any]] = deque()


class UpdateMinerAvailabilitiesForAPI(AsyncLoopRunner):
    miner_availabilities: dict[int, dict] = {}

    async def run_step(self):
        try:
            response = requests.post(
                # TODO check if settings changes are working.
                f"http://{shared_settings.VALIDATOR_API}/miner_availabilities/miner_availabilities",
                headers={"accept": "application/json", "Content-Type": "application/json"},
                json=get_uids(sampling_mode="all"),
                timeout=10,
            )

            self.miner_availabilities = response.json()
        except Exception as e:
            logger.exception(f"Error while updating miner availabilities for API: {e}")
        tracked_availabilities = [m for m in self.miner_availabilities.values() if m is not None]
        logger.debug(
            f"MINER AVAILABILITIES UPDATED, TRACKED: {len(tracked_availabilities)}, UNTRACKED: {len(self.miner_availabilities) - len(tracked_availabilities)}"
        )


update_miner_availabilities_for_api = UpdateMinerAvailabilitiesForAPI()


def filter_available_uids(task: str | None = None, model: str | None = None) -> list[int]:
    """
    Filter UIDs based on task and model availability.

    Args:
        uids: List of UIDs to filter
        task: Task type to check availability for, or None if any task is acceptable
        model: Model name to check availability for, or None if any model is acceptable

    Returns:
        List of UIDs that can serve the requested task/model combination
    """
    filtered_uids = []

    for uid in get_uids(sampling_mode="all"):
        # Skip if miner data is None/unavailable
        if update_miner_availabilities_for_api.miner_availabilities.get(str(uid)) is None:
            continue

        miner_data = update_miner_availabilities_for_api.miner_availabilities[str(uid)]

        # Check task availability if specified
        if task is not None:
            if not miner_data["task_availabilities"].get(task, False):
                continue

        # Check model availability if specified
        if model is not None:
            if not miner_data["llm_model_availabilities"].get(model, False):
                continue

        filtered_uids.append(uid)

    return filtered_uids


# TODO: Modify this so that all the forwarded responses are sent in a single request. This is both more efficient but
# also means that on the validator side all responses are scored at once, speeding up the scoring process.
async def forward_response(uids: list[int], body: dict[str, any], chunks: list[list[str]]):
    uids = [int(u) for u in uids]
    chunk_dict = {u: c for u, c in zip(uids, chunks)}
    logger.info(f"Forwarding response from uid {uids} to scoring with body: {body} and chunks: {chunks}")
    if not shared_settings.SCORE_ORGANICS:
        return

    if body.get("task") != "InferenceTask" and body.get("task") != "WebRetrievalTask":
        logger.debug(f"Skipping forwarding for non- inference/web retrieval task: {body.get('task')}")
        return

    url = f"http://{shared_settings.VALIDATOR_API}/scoring"
    payload = {"body": body, "chunks": chunk_dict, "uid": uids}

    _scoring_queue.append(payload)
    _scoring_lock.popleft()
    try:
        timeout = httpx.Timeout(timeout=120.0, connect=60.0, read=30.0, write=30.0, pool=5.0)
        async with httpx.AsyncClient(timeout=timeout) as client:
            response = await client.post(
                url, json=payload, headers={"api-key": shared_settings.SCORING_KEY, "Content-Type": "application/json"}
            )
            if response.status_code == 200:
                logger.info(f"Forwarding response completed with status {response.status_code}")
            else:
                logger.exception(
                    f"Forwarding response uid {uids} failed with status {response.status_code} and payload {payload}"
                )
    except Exception as e:
        logger.error(f"Tried to forward response to {url} with payload {payload}")
        logger.exception(f"Error while forwarding response: {e}")
