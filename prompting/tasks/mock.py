from prompting.tasks.base_task import BaseTask, WeightedRewardModel
from prompting.rewards.float_diff import FloatDiffModel
from prompting.datasets.base import Context
from prompting.rewards.reward import BaseRewardConfig
from pydantic import model_validator


class MockRewardConfig(BaseRewardConfig):
    reward_definitions: list[WeightedRewardModel] = [WeightedRewardModel(weight=1, reward_model=FloatDiffModel())]


class MockTask(BaseTask):
    static_reference: bool = True
    static_query: bool = True
    context: Context
    reference: str = "This is the reference answer"
    query: str | None = None

    @model_validator(mode="after")
    def make_query(self) -> "MockTask":
        self.query = "How can I solve the following problem, " + self.context.content + "?"
        return self
