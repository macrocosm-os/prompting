import numpy as np
import time
from typing import Literal
from abc import ABC, abstractmethod
from prompting.base.dendrite import DendriteResponseEvent
from pydantic import BaseModel, ConfigDict
from pydantic import model_validator

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
    extra_info: dict
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


# class RewardResult(BaseModel):
#     reward_model: list[BaseRewardModel]
#     agent: BaseLLM
#     response_event: DendriteResponseEvent
#     device: str
#     task_rewards: Optional[list[BaseRewardModel]]
#     task_penalties: Optional[list[BaseRewardModel]]
#     reward_events: Optional[list[RewardEvent]]
#     penalty_events: Optional[list[RewardEvent]]
#     rewards: np.ndarray
#     model_config = ConfigDict(arbitrary_types_allowed=True)

#     @model_validator(mode="after")
#     def compute_rewards(self) -> "RewardResult":
#         self.task_rewards = self.agent.task.reward_definition
#         self.task_penalties = self.agent.task.penalty_definition + self.agent.task.global_penalty_definition
#         self.reward_events = self.reward_responses(
#             reference=self.agent.task.reference,
#             models=self.task_rewards,
#             reward_type="reward",
#         )
#         self.penalty_events = self.reward_responses(
#             reference=self.agent.challenge,
#             models=self.task_penalties,
#             reward_type="penalty",
#         )
#         self.rewards = self.total_reward()

#     def reward_responses(
#         self, reference: str, models: list[BaseRewardModel], reward_type: RewardTypeLiteral
#     ) -> list[RewardEvent]:
#         """Calculates the rewards for the responses given the task and returns a RewardEvent for each reward model
#         reward_events: List[RewardEvent] = [
#             RewardEvent(model_name='rouge', rewards=torch.zeros(50), timings=torch.zeros(50), ...),
#             RewardEvent(model_name='relevance', rewards=torch.zeros(50), timings=torch.zeros(50), ...),
#         ]
#         """
#         reward_events = []

#         for reward_model in models:
#             # Compute the rewards for the responses given the prompt
#             reward_event = reward_model.apply(reference, self.response_event, reward_type=reward_type)
#             reward_events.append(reward_event)

#         return reward_events

#     def total_reward(self) -> np.ndarray:
#         """Combines the rewards from all the reward models into a single reward tensor"""

#         # TODO: How would using the Agent as a reward model fit into this flow?
#         # Compute the rewards for the responses given the prompt
#         rewards = np.zeros_like(self.response_event.uids, dtype=np.float32)

#         for event in self.reward_events:
#             for reward_info in filter(lambda x: x["name"] == event.model_name, self.task_rewards):
#                 rewards += reward_info["weight"] * event.rewards

#         for event in self.penalty_events:
#             for reward_info in filter(lambda x: x["name"] == event.model_name, self.task_penalties):
#                 rewards *= 1 - reward_info["weight"] * event.rewards

#         return rewards

#     def __str__(self):
#         return f"{self.__class__.__name__}(rewards={self.rewards!r}, reward_events={self.reward_events!r}, penalty_events={self.penalty_events!r})"


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

    reward_definitions: list[WeightedRewardModel]
    penalty_definitions: list[WeightedRewardModel] = []

    reward_events: list[WeightedRewardEvent] | None = None
    penalty_events: list[WeightedRewardEvent] | None = None

    @property
    def total_rewards(self) -> list[float]:
        if not self.reward_events:
            raise Exception("Rewards have not yet been calculated")
        return np.sum([r.reward_event.rewards for r in self.reward_events], axis=0)

    @property
    def total_penalties(self) -> list[float]:
        if not self.penalty_events:
            return 0
        return np.sum([r.reward_event.rewards for r in self.penalty_events], axis=0)

    @property
    def final_rewards(self) -> list[float]:
        return self.total_rewards - self.total_penalties

    @model_validator(mode="after")
    def check_summation(self) -> "BaseRewardConfig":
        assert sum([r.weight for r in self.reward_definitions]) == 1, "All rewards must sum to one"

    def apply(self, response_event: DendriteResponseEvent, reference: str, challenge: str | None = None) -> list[float]:
        for weighted_reward in self.reward_definitions:
            self.reward_events = []
            self.reward_events.append(
                WeightedRewardEvent(
                    weight=weighted_reward.weight,
                    reward_event=weighted_reward.reward_model.apply(
                        reference=reference, response_event=response_event, challenge=challenge, reward_type="reward"
                    ),
                )
            )

        if self.penalty_definitions and not challenge:
            raise Exception("You must be providing the challenge to apply penalties")

        for weighted_reward in self.penalty_definitions:
            self.penalty_events = []
            self.penalty_events.append(
                WeightedRewardEvent(
                    weight=weighted_reward.weight,
                    reward_event=weighted_reward.reward_model.apply(
                        reference=challenge, response_event=response_event, reward_type="penalty"
                    ),
                )
            )
        return self.final_rewards
