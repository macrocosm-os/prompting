from prompting.rewards.exact_match import LogitsRewardModel
from prompting.rewards.relevance import RelevanceRewardModel
from prompting.rewards.reward import BaseRewardModel, BatchRewardOutput
from prompting.tasks.base_task import BaseTextTask
from shared.dendrite import DendriteResponseEvent


class InferenceRewardModel(BaseRewardModel):
    async def reward(
        self,
        reference: str,
        response_event: DendriteResponseEvent,
        model_id: str | None = None,
        task: BaseTextTask | None = None,
        **kwargs,
    ) -> BatchRewardOutput:
        """Gives an exact reward of 1 if the response matches the reference, 0 otherwise"""
        if model_id:
            return await LogitsRewardModel().reward(reference, response_event, task)
        return await RelevanceRewardModel().reward(reference, response_event)
