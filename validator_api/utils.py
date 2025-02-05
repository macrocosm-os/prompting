import requests
from loguru import logger

from shared.loop_runner import AsyncLoopRunner
from shared.settings import shared_settings
from shared.uids import get_uids


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
