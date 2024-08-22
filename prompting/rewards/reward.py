import numpy as np
import time
from typing import Literal, ClassVar
from abc import ABC, abstractmethod
from prompting.base.dendrite import DendriteResponseEvent
from pydantic import BaseModel, ConfigDict

RewardTypeLiteral = Literal["reward", "penalty"]


class RewardEvent(BaseModel):
    """Contains rewards for all the responses in a batch"""

    model_name: str
    rewards: np.ndarray
    rewards_normalized: np.ndarray
    timings: np.ndarray
    model_type: RewardTypeLiteral
    batch_time: float
    extra_info: dict
    model_config = ConfigDict(arbitrary_types_allowed=True)

    # implement custom asdict to return a dict with the same keys as the dataclass using the model name
    def asdict(self) -> dict:
        return {
            f"{self.model_name}_raw_{self.model_type.value}": self.tensor_to_rounded_list(self.rewards),
            f"{self.model_name}_{self.model_type.value}": self.tensor_to_rounded_list(self.rewards_normalized, 4),
            f"{self.model_name}_{self.model_type.value}_timings": self.tensor_to_rounded_list(self.timings),
            f"{self.model_name}_{self.model_type.value}_batch_time": self.batch_time,
            f"{self.model_name}_{self.model_type.value}_extra_info": self.extra_info,
        }

    def tensor_to_rounded_list(self, tensor, decimals=6):
        # Convert the tensor elements to floats and round them to 6 decimal places
        return [round(float(element), decimals) for element in tensor]


class BatchRewardOutput(BaseModel):
    rewards: np.ndarray
    timings: np.ndarray
    extra_info: dict = {}
    model_config = ConfigDict(arbitrary_types_allowed=True)

    @property
    def rewards_normalized(self) -> np.ndarray:
        return self.rewards / sum(self.rewards)

    def __post_init__(self):
        if self.rewards.shape != self.timings.shape:
            raise ValueError(f"rewards.shape {self.rewards.shape} != timings.shape {self.timings.shape}")

        self.rewards_normalized = (self.rewards - self.rewards.min()) / (self.rewards.max() - self.rewards.min() + 1e-6)


class BaseRewardModel(ABC, BaseModel):
    @abstractmethod
    def reward(self, reference: str, response_event: DendriteResponseEvent) -> BatchRewardOutput:
        pass

    def apply(
        self,
        response_event: DendriteResponseEvent,
        reference: str | None = None,
        challenge: str | None = None,
        reward_type: Literal["reward", "penalty"] = "reward",
    ) -> RewardEvent:
        t0 = time.time()
        comparator = reference if reward_type == "reward" else challenge
        batch_rewards_output: BatchRewardOutput = self.reward(comparator, response_event)
        batch_rewards_time = time.time() - t0

        return RewardEvent(
            model_name=self.__class__.__name__,
            rewards=batch_rewards_output.rewards,
            rewards_normalized=batch_rewards_output.rewards_normalized,
            model_type=reward_type,
            batch_time=batch_rewards_time,
            extra_info=batch_rewards_output.extra_info,
            timings=batch_rewards_output.timings,
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
        cls, response_event: DendriteResponseEvent, reference: str, challenge: str | None = None
    ) -> tuple[list[WeightedRewardEvent], list[WeightedRewardEvent], list[float]]:
        reward_events = []
        for weighted_reward in cls.reward_definitions:
            reward_events.append(
                WeightedRewardEvent(
                    weight=weighted_reward.weight,
                    reward_event=weighted_reward.reward_model.apply(
                        reference=reference, response_event=response_event, challenge=challenge, reward_type="reward"
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
