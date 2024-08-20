from pydantic import BaseModel
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

task_config = dict[str, bool] = {
    DateQuestionAnsweringTask.__name__: False,
    QuestionAnsweringTask.__name__: False,
    SummarizationTask.__name__: False,
    SyntheticInferenceTask.__name__: False,
    BaseInferenceTask.__name__: False,
}
model_config: dict[str, bool] = [{conf.model_id: False} for conf in ModelZoo.models_configs]


class MinerAvailability(BaseModel):
    """This class keeps track of one miner's availability"""

    task_availabilities: dict[str, bool] = task_config
    model_availabilities: list[dict] = model_config

    def is_model_available(self, model: str) -> bool:
        return self.model_availabilities[model]

    def is_task_available(self, task: BaseTask) -> bool:
        return self.task_availabilities[task.__class__.__name__]


class MinerAvailabilities(BaseModel):
    """This class keeps track of all the miner's availabilities and
    let's us target a miner based on its availability"""

    miners: dict[int, MinerAvailability] = {}

    def available_miners_by_model(model: str) -> list[str]:
        return [uid for uid, miner in miner_availabilities.miners.items() if miner.is_model_available(model)]

    def available_miners_by_task(task: BaseTask) -> list[str]:
        return [uid for uid, miner in miner_availabilities.miners.items() if miner.is_task_available(task)]


class CheckMinerAvailability(AsyncLoopRunner):
    interval: int = 60

    async def run_step(self):
        uids = get_uids(sampling_mode="all")
        axons = [settings.METAGRAPH.axons[uid] for uid in uids]
        responses: list[AvailabilitySynapse] = await settings.DENDRITE(
            axons=axons,
            synapse=AvailabilitySynapse(task_config=task_config, model_config=model_config),
            timeout=settings.NEURON_TIMEOUT,
            deserialize=False,
            streaming=True,
        )
        for response, uid in zip(responses, uids):
            miner_availabilities.miners[uid] = MinerAvailability(
                task_availabilities=response.task_availabilities,
                model_availabilities=response.model_availabilities,
            )


miner_availabilities = MinerAvailabilities()
checking_loop = CheckMinerAvailability()
checking_loop.start()


# Example usage
# miner_availabilities.available_miners_by_model("gpt-neo-2.7B")
# miner_availabilities.available_miners_by_task(DateQuestionAnsweringTask())
