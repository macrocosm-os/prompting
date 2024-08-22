from pydantic import BaseModel
from loguru import logger
from prompting.tasks.base_task import BaseTask
from prompting.llms.model_zoo import ModelZoo
from prompting.base.loop_runner import AsyncLoopRunner
from prompting.base.protocol import AvailabilitySynapse
from prompting.settings import settings
from prompting.tasks.date_qa import DateQuestionAnsweringTask
from prompting.tasks.qa import QuestionAnsweringTask
from prompting.tasks.summarization import SummarizationTask
from prompting.tasks.inference import SyntheticInferenceTask, BaseInferenceTask
from prompting.utils.uids import get_uids
import random

task_config: dict[str, bool] = {
    DateQuestionAnsweringTask.__name__: False,
    QuestionAnsweringTask.__name__: False,
    SummarizationTask.__name__: False,
    SyntheticInferenceTask.__name__: False,
    BaseInferenceTask.__name__: False,
}
model_config: dict[str, bool] = {conf.model_id: False for conf in ModelZoo.models_configs}


class MinerAvailability(BaseModel):
    """This class keeps track of one miner's availability"""

    task_availabilities: dict[str, bool] = task_config
    model_availabilities: dict[str, bool] = model_config

    def is_model_available(self, model: str) -> bool:
        return self.model_availabilities[model]

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
        axons = [settings.METAGRAPH.axons[uid] for uid in uids]
        availability_synapse = AvailabilitySynapse(task_availabilities=task_config, model_availabilities=model_config)
        responses: list[AvailabilitySynapse] = await settings.DENDRITE(
            axons=axons,
            synapse=availability_synapse,
            timeout=settings.NEURON_TIMEOUT,
            # timeout=100,
            deserialize=False,
            streaming=False,
        )
        logger.debug(f"MINER AVAILABILITY RESPONSES: {responses}")
        for response, uid in zip(responses, uids):
            miner_availabilities.miners[uid] = MinerAvailability(
                task_availabilities=response.task_availabilities,
                model_availabilities=response.model_availabilities,
            )
        logger.debug(f"MINER AVAILABILITIES: {miner_availabilities.miners}")


miner_availabilities = MinerAvailabilities()
checking_loop = CheckMinerAvailability()
