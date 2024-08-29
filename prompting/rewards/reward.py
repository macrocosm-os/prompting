import numpy as np
import time
from typing import Literal, ClassVar
from abc import ABC, abstractmethod
from prompting.base.dendrite import DendriteResponseEvent
from pydantic import BaseModel, ConfigDict

RewardTypeLiteral = Literal["reward", "penalty"]


class RewardEvent(BaseModel):
    """Contains rewards for all the responses in a batch"""

    reward_model_name: str
    rewards: np.ndarray
    rewards_normalized: np.ndarray
    timings: np.ndarray
    reward_model_type: RewardTypeLiteral
    batch_time: float
    threshold: float | None = None
    extra_info: dict | None = None

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
    @abstractmethod
    def reward(self, reference: str, response_event: DendriteResponseEvent) -> BatchRewardOutput:
        raise NotImplementedError("You must implement the reward method")

    def apply(
        self,
        response_event: DendriteResponseEvent,
        reference: str | None = None,
        challenge: str | None = None,
        reward_type: Literal["reward", "penalty"] = "reward",
        **kwargs,
    ) -> RewardEvent:
        t0 = time.time()
        comparator = reference if reward_type == "reward" else challenge
        batch_rewards_output: BatchRewardOutput = self.reward(comparator, response_event)
        batch_rewards_time = time.time() - t0

        return RewardEvent(
            reward_model_name=self.__class__.__name__,
            rewards=batch_rewards_output.rewards,
            rewards_normalized=batch_rewards_output.rewards_normalized,
            reward_model_type=reward_type,
            batch_time=batch_rewards_time,
            threshold=batch_rewards_output.threshold,
            timings=batch_rewards_output.timings,
            extra_info=kwargs,
        )


class WeightedRewardModel(BaseModel):
    weight: float
    reward_model: BaseRewardModel


class WeightedRewardEvent(BaseModel):
    weight: float
    reward_event: RewardEvent


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

    reward_definitions: ClassVar[list[WeightedRewardModel]]
    penalty_definitions: ClassVar[list[WeightedRewardModel]] = []

    @classmethod
    def sum_rewards(cls, reward_events: list[WeightedRewardEvent]) -> list[float]:
        if not reward_events:
            return 0
        return np.sum([r.reward_event.rewards for r in reward_events], axis=0)

    @classmethod
    def final_rewards(
        cls, reward_events: list[WeightedRewardEvent], penalty_events: list[WeightedRewardEvent]
    ) -> list[float]:
        return cls.sum_rewards(reward_events) - cls.sum_rewards(penalty_events)

    @classmethod
    def apply(
        cls,
        response_event: DendriteResponseEvent,
        reference: str,
        challenge: str | None = None,
        model_id: str | None = None,
    ) -> tuple[list[WeightedRewardEvent], list[WeightedRewardEvent], list[float]]:
        reward_events = []
        for weighted_reward in cls.reward_definitions:
            reward_events.append(
                WeightedRewardEvent(
                    weight=weighted_reward.weight,
                    reward_event=weighted_reward.reward_model.apply(
                        reference=reference,
                        response_event=response_event,
                        challenge=challenge,
                        reward_type="reward",
                        model_id=model_id,
                    ),
                )
            )

        if cls.penalty_definitions and not challenge:
            raise Exception("You must be providing the challenge to apply penalties")

        penalty_events = []
        for weighted_reward in cls.penalty_definitions:
            penalty_events.append(
                WeightedRewardEvent(
                    weight=weighted_reward.weight,
                    reward_event=weighted_reward.reward_model.apply(
                        reference=challenge, response_event=response_event, reward_type="penalty"
                    ),
                )
            )
        return reward_events, penalty_events, cls.final_rewards(reward_events, penalty_events)
