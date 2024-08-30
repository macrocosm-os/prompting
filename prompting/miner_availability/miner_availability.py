from pydantic import BaseModel
from loguru import logger
from prompting.tasks.base_task import BaseTask
from prompting.llms.model_zoo import ModelZoo
from prompting.base.loop_runner import AsyncLoopRunner
from prompting.base.protocol import AvailabilitySynapse
from prompting.settings import settings
from prompting.tasks.task_registry import TaskRegistry
from prompting.utils.uids import get_uids
import random

task_config = dict[str, bool] = {task_config.task.__name__: True for task_config in TaskRegistry.task_configs}
# task_config: dict[str, bool] = {
#     DateQuestionAnsweringTask.__name__: True,
#     QuestionAnsweringTask.__name__: True,
#     SummarizationTask.__name__: True,
#     SyntheticInferenceTask.__name__: True,
#     OrganicInferenceTask.__name__: True,
# }
model_config: dict[str, bool] = {conf.llm_model_id: False for conf in ModelZoo.models_configs}


class MinerAvailability(BaseModel):
    """This class keeps track of one miner's availability"""

    task_availabilities: dict[str, bool] = task_config
    llm_model_availabilities: dict[str, bool] = model_config

    def is_model_available(self, model: str) -> bool:
        return self.llm_model_availabilities[model]

    def is_task_available(self, task: BaseTask) -> bool:
        return self.task_availabilities[task.__class__.__name__]


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
        return available


class CheckMinerAvailability(AsyncLoopRunner):
    interval: int = 10

    async def run_step(self):
        uids = settings.TEST_MINER_IDS or get_uids(sampling_mode="all")
        logger.info(f"Collecting miner availabilities on uids: {uids}")
        if any([len(settings.METAGRAPH.axons) <= uid for uid in uids]):
            raise Exception("Some UIDs are out of bounds. Make sure all the TEST_MINER_IDS are valid.")
        axons = [settings.METAGRAPH.axons[uid] for uid in uids]
        responses: list[AvailabilitySynapse] = await settings.DENDRITE(
            axons=axons,
            synapse=AvailabilitySynapse(task_availabilities=task_config, llm_model_availabilities=model_config),
            timeout=settings.NEURON_TIMEOUT,
            deserialize=False,
            streaming=False,
        )
        for response, uid in zip(responses, uids):
            miner_availabilities.miners[uid] = MinerAvailability(
                task_availabilities=response.task_availabilities,
                llm_model_availabilities=response.llm_model_availabilities,
            )
        logger.debug("Miner availabilities updated.")


miner_availabilities = MinerAvailabilities()
availability_checking_loop = CheckMinerAvailability()
