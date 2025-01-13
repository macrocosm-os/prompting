import asyncio
import random
from typing import Dict

import numpy as np
from loguru import logger
from pydantic import BaseModel

from prompting.llms.model_zoo import ModelZoo
from prompting.tasks.base_task import BaseTask
from prompting.tasks.task_registry import TaskRegistry
from shared.epistula import query_availabilities
from shared.loop_runner import AsyncLoopRunner
from shared.settings import shared_settings
from shared.uids import get_uids

task_config: dict[str, bool] = {str(task_config.task.__name__): True for task_config in TaskRegistry.task_configs}
model_config: dict[str, bool] = {conf.llm_model_id: False for conf in ModelZoo.models_configs}


class MinerAvailability(BaseModel):
    """This class keeps track of one miner's availability"""

    task_availabilities: dict[str, bool] = task_config
    llm_model_availabilities: dict[str, bool] = model_config

    def is_model_available(self, model: str) -> bool:
        return self.llm_model_availabilities.get(model, False)

    def is_task_available(self, task: BaseTask | type[BaseTask]) -> bool:
        if isinstance(task, BaseTask):
            return self.task_availabilities.get(task.__class__.__name__, False)
        return self.task_availabilities.get(task.__name__, False)


class MinerAvailabilities(BaseModel):
    """This class keeps track of all the miner's availabilities and
    let's us target a miner based on its availability"""

    miners: dict[int, MinerAvailability] = {}

    def get_available_miners(
        self, task: BaseTask | None = None, model: str | None = None, k: int | None = None
    ) -> list[int]:
        available = list(self.miners.keys())
        if task:
            available = [uid for uid in available if self.miners[uid].is_task_available(task)]
        if model:
            available = [uid for uid in available if self.miners[uid].is_model_available(model)]
        if k:
            available = random.sample(available, min(len(available), k))
        return list(map(int, available))


class CheckMinerAvailability(AsyncLoopRunner):
    interval: int = 30  # Miners will be queried approximately once every hour
    uids: np.ndarray = shared_settings.TEST_MINER_IDS or get_uids(sampling_mode="all")
    current_index: int = 0
    uids_per_step: int = 10

    class Config:
        arbitrary_types_allowed = True

    async def run_step(self):
        start_index = self.current_index
        end_index = min(start_index + self.uids_per_step, len(self.uids))
        uids_to_query = self.uids[start_index:end_index]
        if self.step == 0:
            uids_to_query = self.uids

        if any(uid >= len(shared_settings.METAGRAPH.axons) for uid in uids_to_query):
            raise ValueError("Some UIDs are out of bounds. Make sure all the TEST_MINER_IDS are valid.")
        responses: list[Dict[str, bool]] = await query_availabilities(uids_to_query, task_config, model_config)

        for response, uid in zip(responses, uids_to_query):
            try:
                miner_availabilities.miners[uid] = MinerAvailability(
                    task_availabilities=response["task_availabilities"],
                    llm_model_availabilities=response["llm_model_availabilities"],
                )
            except Exception:
                logger.debug("Availability Response Invalid")
                miner_availabilities.miners[uid] = MinerAvailability(
                    task_availabilities={task: True for task in task_config},
                    llm_model_availabilities={model: False for model in model_config},
                )

        logger.debug("Miner availabilities updated.")
        self.current_index = end_index

        if self.current_index >= len(self.uids):
            self.current_index = 0

        await asyncio.sleep(0.1)


miner_availabilities = MinerAvailabilities()
availability_checking_loop = CheckMinerAvailability()
