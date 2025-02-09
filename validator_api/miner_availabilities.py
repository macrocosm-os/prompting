import random
from typing import Optional

import requests
from fastapi import APIRouter
from loguru import logger
from pydantic import BaseModel

from shared.loop_runner import AsyncLoopRunner
from shared.settings import shared_settings
from shared.uids import get_uids

router = APIRouter()


# Move the class definition before its usage
class APIMinerAvailability(BaseModel):
    task_availabilities: dict[str, bool]
    llm_model_availabilities: dict[str, bool]


# Initialize as a dict with the correct type annotation
miner_availabilities: dict[str, APIMinerAvailability] = {}


def get_available_miner(task: Optional[str] = None, model: Optional[str] = None) -> Optional[str]:
    """
    Fetch an available miner that supports the specified task and model.

    Args:
        task (Optional[str]): The task to check for. Defaults to "InferenceTask" if None
        model (Optional[str]): The model to check for. No constraints if None

    Returns:
        Optional[str]: UID of the first available miner that meets the criteria, or None if no miner is available
    """
    task = task or "InferenceTask"

    valid_uids = []
    for uid, availability in miner_availabilities.items():
        # Check if the miner supports the required task
        if not availability.task_availabilities.get(task, False):
            continue

        # If model is specified, check if the miner supports it
        if model is not None:
            if not availability.llm_model_availabilities.get(model, False):
                continue

        # If we reach here, the miner meets all requirements
        valid_uids.append(uid)
    if valid_uids:
        return int(random.choice(valid_uids))

    # No suitable miner found
    logger.warning(f"No available miner found for task: {task}, model: {model}")
    return None


class MinerAvailabilitiesUpdater(AsyncLoopRunner):
    interval: int = 40

    async def run_step(self):
        uids = get_uids(sampling_mode="random", k=100) # TODO: We should probably not just randomly sample uids, there's likely a better way to do this.
        # TODO: Default to highest stake validator's availability api
        url = f"{shared_settings.VALIDATOR_API}/miner_availabilities/miner_availabilities"

        try:
            # TODO: Need to add some level of ddos protection for this
            result = requests.post(url, json=uids.tolist(), timeout=10)
            result.raise_for_status()  # Raise an exception for bad status codes

            response_data = result.json()

            # Clear existing availabilities before updating
            miner_availabilities.clear()

            # Update availabilities for each UID
            for uid, miner_availability in response_data.items():
                if miner_availability is not None:  # Skip null values
                    try:
                        miner_availabilities[uid] = APIMinerAvailability(**miner_availability)
                    except Exception as e:
                        logger.error(f"Failed to parse miner availability for UID {uid}: {str(e)}")
            logger.debug(
                f"Updated miner availabilities for {len(miner_availabilities)} miners, sample availabilities: {list(miner_availabilities.items())[:2]}"
            )

        except requests.exceptions.ConnectionError as e:
            logger.error(
                f"Failed to update miner availabilities - check that your validator is running and all ports/ips are set correctly!: {e}"
            )
        except requests.exceptions.RequestException as e:
            logger.error(f"Failed to update miner availabilities: {e}")
        except Exception as e:
            logger.exception(f"Unexpected error while updating miner availabilities: {e}")


availability_updater = MinerAvailabilitiesUpdater()
