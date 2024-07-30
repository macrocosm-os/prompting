import torch
import time
from typing import Optional, Literal
from abc import ABC, abstractmethod
from prompting.dendrite import DendriteResponseEvent
from pydantic import BaseModel, ConfigDict
from prompting.llms.base_llm import BaseLLM
from pydantic import model_validator

RewardTypeLiteral = Literal["reward", "penalty"]


class RewardEvent(BaseModel):
    """Contains rewards for all the responses in a batch"""

    model_name: str
    rewards: torch.FloatTensor
    rewards_normalized: torch.FloatTensor
    timings: torch.FloatTensor
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
    rewards: torch.FloatTensor
    timings: torch.FloatTensor
    extra_info: dict
    model_config = ConfigDict(arbitrary_types_allowed=True)

    @property
    def rewards_normalized(self) -> torch.FloatTensor:
        return self.rewards / sum(self.rewards)

    def __post_init__(self):
        if self.rewards.shape != self.timings.shape:
            raise ValueError(f"rewards.shape {self.rewards.shape} != timings.shape {self.timings.shape}")

        self.rewards_normalized = (self.rewards - self.rewards.min()) / (self.rewards.max() - self.rewards.min() + 1e-6)


class BaseRewardModel(ABC, BaseModel):
    @property
    @abstractmethod
    def name(self) -> str: ...

    @abstractmethod
    def reward(self, reference: str, response_event: DendriteResponseEvent) -> BatchRewardOutput:
        pass

    def apply(
        self, reference: str, response_event: DendriteResponseEvent, reward_type: Literal["reward", "penalty"]
    ) -> RewardEvent:
        t0 = time.time()
        batch_rewards_output: BatchRewardOutput = self.reward(reference, response_event)
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

    def __repr__(self):
        return f"{self.__class__.__name__}(name={self.name})"


class RewardResult(BaseModel):
    reward_pipeline: list[BaseRewardModel]
    agent: BaseLLM
    response_event: DendriteResponseEvent
    device: str
    task_rewards: Optional[list[BaseRewardModel]]
    task_penalties: Optional[list[BaseRewardModel]]
    reward_events: Optional[list[RewardEvent]]
    penalty_events: Optional[list[RewardEvent]]
    rewards: torch.FloatTensor
    model_config = ConfigDict(arbitrary_types_allowed=True)

    @model_validator(mode="after")
    def compute_rewards(self) -> "RewardResult":
        self.task_rewards = self.agent.task.reward_definition
        self.task_penalties = self.agent.task.penalty_definition + self.agent.task.global_penalty_definition
        self.reward_events = self.reward_responses(
            reference=self.agent.task.reference,
            models=self.task_rewards,
            reward_type="reward",
        )
        self.penalty_events = self.reward_responses(
            reference=self.agent.challenge,
            models=self.task_penalties,
            reward_type="penalty",
        )
        self.rewards = self.total_reward()

    def reward_responses(
        self, reference: str, models: list[BaseRewardModel], reward_type: RewardTypeLiteral
    ) -> list[RewardEvent]:
        """Calculates the rewards for the responses given the task and returns a RewardEvent for each reward model
        reward_events: List[RewardEvent] = [
            RewardEvent(model_name='rouge', rewards=torch.zeros(50), timings=torch.zeros(50), ...),
            RewardEvent(model_name='relevance', rewards=torch.zeros(50), timings=torch.zeros(50), ...),
        ]
        """
        reward_events = []

        for reward_model in models:
            # Compute the rewards for the responses given the prompt
            reward_event = reward_model.apply(reference, self.response_event, reward_type=reward_type)
            reward_events.append(reward_event)

        return reward_events

    def total_reward(self) -> torch.FloatTensor:
        """Combines the rewards from all the reward models into a single reward tensor"""

        # TODO: How would using the Agent as a reward model fit into this flow?
        # Compute the rewards for the responses given the prompt
        rewards = torch.zeros_like(self.response_event.uids, dtype=torch.float32, device=self.device)

        for event in self.reward_events:
            for reward_info in filter(lambda x: x["name"] == event.model_name, self.task_rewards):
                rewards += reward_info["weight"] * event.rewards.to(self.device)

        for event in self.penalty_events:
            for reward_info in filter(lambda x: x["name"] == event.model_name, self.task_penalties):
                rewards *= 1 - reward_info["weight"] * event.rewards.to(self.device)

        return rewards

    def __str__(self):
        return f"{self.__class__.__name__}(rewards={self.rewards!r}, reward_events={self.reward_events!r}, penalty_events={self.penalty_events!r})"
