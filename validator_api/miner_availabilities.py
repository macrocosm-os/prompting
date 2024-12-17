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


class MinerAvailabilitiesUpdater(AsyncLoopRunner):
    interval: int = 10

    async def run_step(self):
        uids = get_uids(sampling_mode="all")
        url = f"http://{shared_settings.VALIDATOR_IP}:{shared_settings.VALIDATOR_PORT}/miner_availabilities/miner_availabilities"

        try:
            result = requests.post(url, json=uids)
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

        except requests.exceptions.RequestException as e:
            logger.exception(f"Failed to update miner availabilities: {e}")
        except Exception as e:
            logger.exception(f"Unexpected error while updating miner availabilities: {e}")


availability_updater = MinerAvailabilitiesUpdater()
