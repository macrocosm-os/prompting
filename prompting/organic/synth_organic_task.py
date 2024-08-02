from dataclasses import dataclass

from prompting.organic.organic_task import OrganicTask
from prompting.rewards.rouge import RougeRewardModel
from prompting.rewards.relevance import RelevanceRewardModel
from prompting.rewards.reward import WeightedRewardModel


@dataclass
class SynthOrganicTask(OrganicTask):
    """Task with defined reward and penalty mechanisms for synthetic organic prompts."""

    name = "synthetic-organic"
    reward_definitions: list[WeightedRewardModel] = [
        WeightedRewardModel(weight=0.5, reward_model=RougeRewardModel()),
        WeightedRewardModel(weight=0.5, reward_model=RelevanceRewardModel()),
    ]
    penalty_definition: list[WeightedRewardModel] = [
        WeightedRewardModel(weight=0.5, reward_model=RelevanceRewardModel())
    ]

    cleaning_pipeline = []

    def __init__(self, context: dict, reference: str):
        self.context = context
        self.messages = context["messages"]
        self.roles = context["roles"]
        self.query = context["messages"][-1]
        self.topic = "Organic"
        self.reference = reference
        self.subtopic = ""
        self.tags = [""]
