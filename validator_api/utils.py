import random

import requests
from loguru import logger

from shared import settings
from shared.loop_runner import AsyncLoopRunner
from shared.uids import get_uids


def read_fallback_uids() -> dict[str, dict]:
    try:
        from collections import defaultdict

        uids = get_uids(sampling_mode="all")
        return {
            str(uid): {
                "task_availabilities": defaultdict(lambda: True),
                "llm_model_availabilities": defaultdict(lambda: True),
            }
            for uid in uids
        }
    except Exception as e2:
        logger.error(f"Error reading miner availabilities from JSON file: {e2}")
        return {}


class UpdateMinerAvailabilitiesForAPI(AsyncLoopRunner):
    interval: int = 120
    miner_availabilities: dict[int, dict] = {}

    async def run_step(self):
        logger.debug("Running update miner availabilities step")
        if settings.shared_settings.API_TEST_MODE:
            return
        try:
            response = requests.post(
                f"http://{settings.shared_settings.VALIDATOR_API}/miner_availabilities/miner_availabilities",
                headers={"accept": "application/json", "Content-Type": "application/json"},
                json=get_uids(sampling_mode="all"),
                timeout=15,
            )
            self.miner_availabilities = response.json()
        except Exception as e:
            logger.error(f"Failed updating miner availabilities for API, fallback to all uids: {e}")
            self.miner_availabilities = read_fallback_uids()
        tracked_availabilities = [m for m in self.miner_availabilities.values() if m is not None]
        logger.info(f"Availabilities updated, tracked: {len(tracked_availabilities)}")


update_miner_availabilities_for_api = UpdateMinerAvailabilitiesForAPI()


def filter_available_uids(
    task: str | None = None,
    model: str | None = None,
    test: bool = False,
    n_miners: int = 10,
    n_top_incentive: int = 100,
) -> list[int]:
    """Filter UIDs based on task and model availability.

    Args:
        task (str | None, optional): The task to filter miners by. Defaults to None.
        model (str | None, optional): The LLM model to filter miners by. Defaults to None.
        test (bool, optional): Whether to run in test mode. Defaults to False.
        n_miners (int, optional): Number of miners to return. Defaults to 10.
        n_top_incentive (int, optional): Number of top incentivized miners to consider. Defaults to 10.

    Returns:
        list[int]: List of filtered UIDs that match the criteria.
    """
    if test:
        return get_uids(sampling_mode="top_incentive", k=n_miners)

    filtered_uids = []

    for uid in get_uids(sampling_mode="top_incentive", k=max(n_top_incentive, n_miners)):
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

    if len(filtered_uids) == 0:
        logger.error(
            "Got an empty list of available UIDs, falling back to all uids. "
            "Check VALIDATOR_API and SCORING_KEY in .env.api"
        )
        filtered_uids = get_uids(sampling_mode="top_incentive", k=n_miners)

    filtered_uids = random.sample(filtered_uids, min(len(filtered_uids), n_miners))

    return filtered_uids
