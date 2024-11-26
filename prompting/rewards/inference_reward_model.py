from prompting.base.dendrite import DendriteResponseEvent
from prompting.rewards.exact_match import ExactMatchRewardModel
from prompting.rewards.relevance import RelevanceRewardModel
from prompting.rewards.reward import BaseRewardModel, BatchRewardOutput


class InferenceRewardModel(BaseRewardModel):
    def reward(
        self, reference: str, response_event: DendriteResponseEvent, model_id: str | None = None
    ) -> BatchRewardOutput:
        """Gives an exact reward of 1 if the response matches the reference, 0 otherwise"""
        if model_id:
            return ExactMatchRewardModel().reward(reference, response_event)
        return RelevanceRewardModel().reward(reference, response_event)
