import numpy as np
import time
from typing import Literal, ClassVar
from abc import ABC, abstractmethod
from prompting.base.dendrite import DendriteResponseEvent
from pydantic import BaseModel, ConfigDict
from prompting.tasks.base_task import BaseTextTask

RewardTypeLiteral = Literal["reward", "penalty"]


class WeightedRewardEvent(BaseModel):
    weight: float
    task: BaseTextTask
    reward_model_name: str
    rewards: np.ndarray
    rewards_normalized: np.ndarray
    timings: np.ndarray
    reward_model_type: RewardTypeLiteral
    batch_time: float
    uids: list[float]

    threshold: float | None = None
    extra_info: dict | None = None
    reward_type: Literal["reward", "penalty"] = "reward"

    model_config = ConfigDict(arbitrary_types_allowed=True)

    # implement custom asdict to return a dict with the same keys as the dataclass using the model name
    def asdict(self) -> dict:
        return {
            f"{self.reward_model_name}_raw_{self.reward_model_type.value}": self.rewards,
            f"{self.reward_model_name}_{self.reward_model_type.value}": self.rewards_normalized,
            f"{self.reward_model_name}_{self.reward_model_type.value}_timings": self.timings,
            f"{self.reward_model_name}_{self.reward_model_type.value}_batch_time": self.batch_time,
            f"{self.reward_model_name}_{self.reward_model_type.value}_threshold": self.threshold,
            f"{self.reward_model_name}_{self.reward_model_type.value}_extra_info": self.extra_info,
            f"{self.reward_model_name}_{self.reward_model_type.value}_uids": self.uids,
            f"{self.reward_model_name}_{self.reward_model_type.value}_task": self.task,
            f"{self.reward_model_name}_{self.reward_model_type.value}_weight": self.weight,
        }


class BatchRewardOutput(BaseModel):
    rewards: np.ndarray
    timings: np.ndarray
    threshold: float | None = None
    extra_info: dict = {}
    model_config = ConfigDict(arbitrary_types_allowed=True)

    @property
    def rewards_normalized(self) -> np.ndarray:
        if self.rewards.shape != self.timings.shape:
            raise ValueError(f"rewards.shape {self.rewards.shape} != timings.shape {self.timings.shape}")
        if self.rewards.min() == self.rewards.max():
            return np.array([1 / len(self.rewards)] * len(self.rewards))
        return (self.rewards - self.rewards.min()) / (self.rewards.max() - self.rewards.min())


class BaseRewardModel(ABC, BaseModel):
    weight: float = 1.0

    @abstractmethod
    def reward(self, reference: str, response_event: DendriteResponseEvent) -> BatchRewardOutput:
        raise NotImplementedError("You must implement the reward method")

    def apply(
        self,
        response_event: DendriteResponseEvent,
        reference: str | None = None,
        challenge: str | None = None,
        reward_type: Literal["reward", "penalty"] = "reward",
        task: BaseTextTask | None = None,
        **kwargs,
    ) -> WeightedRewardEvent:
        t0 = time.time()
        comparator = reference if reward_type == "reward" else challenge
        batch_rewards_output: BatchRewardOutput = self.reward(comparator, response_event)
        batch_rewards_time = time.time() - t0

        return WeightedRewardEvent(
            weight=self.weight,
            task=task,
            reward_model_name=self.__class__.__name__,
            rewards=batch_rewards_output.rewards,
            rewards_normalized=batch_rewards_output.rewards_normalized,
            reward_model_type=reward_type,
            batch_time=batch_rewards_time,
            threshold=batch_rewards_output.threshold,
            timings=batch_rewards_output.timings,
            extra_info=kwargs,
            uids=response_event.uids,
        )


class WeightedRewardModel(BaseModel):
    weight: float
    reward_model: BaseRewardModel


class BaseRewardConfig(ABC, BaseModel):
    """This class takes in a dictionary of rewards and penalties that should be applied. On apply(),
    it then applies all the reward models based on query & reference and returns the reward.

    both reward_definition and penalty_definition must be a list of tuples of type:

    weighting: RewardModel, e.g.

    [ (0.2, RougeRewardModel), (0.8, CosineDistanceRewardModel) ]

    Note that for all the rewards, the percentages must sum up to 1 (100%). For penalties,
    this is not the case, e.g. you may want to only apply a single penalty very lightly
    and weight it with <1.
    """

    reward_definitions: ClassVar[list[BaseRewardModel]]
    penalty_definitions: ClassVar[list[BaseRewardModel]] = []

    @classmethod
    def sum_rewards(cls, reward_events: list[WeightedRewardEvent]) -> np.ndarray:
        if not reward_events:
            return 0
        return np.sum([r.rewards * r.weight for r in reward_events], axis=0)

    @classmethod
    def final_rewards(cls, reward_events: list[WeightedRewardEvent]) -> list[float]:
        penalty_events = [r for r in reward_events if r.reward_type == "penalty"]
        reward_events = [r for r in reward_events if r.reward_type == "reward"]
        return cls.sum_rewards(reward_events) - cls.sum_rewards(penalty_events)

    @classmethod
    def apply(
        cls,
        response_event: DendriteResponseEvent,
        reference: str,
        challenge: str | None = None,
        model_id: str | None = None,
        task: BaseTextTask | None = None,
    ) -> list[BaseRewardModel]:
        reward_events: list[BaseRewardModel] = []
        for weighted_reward in cls.reward_definitions:
            reward_events.append(
                weighted_reward.apply(
                    reference=reference,
                    response_event=response_event,
                    challenge=challenge,
                    reward_type="reward",
                    model_id=model_id,
                    task=task,
                ),
            )

        if cls.penalty_definitions and not challenge:
            raise Exception("You must be providing the challenge to apply penalties")

        for weighted_reward in cls.penalty_definitions:
            reward_events.append(
                weighted_reward.apply(
                    reference=challenge,
                    response_event=response_event,
                    reward_type="penalty",
                    task=task,
                ),
            )
        return reward_events
