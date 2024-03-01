import torch
import time
import bittensor as bt
from typing import List
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum


class RewardModelTypeEnum(Enum):
    WEIGHTED_REWARD = "reward"
    FILTER_REWARD = "filter"
    PENALTY = "penalty"


@dataclass
class RewardEvent:
    """Contains rewards for all the responses in a batch"""

    model_name: str
    rewards: torch.FloatTensor
    rewards_normalized: torch.FloatTensor
    timings: torch.FloatTensor
    model_type: RewardModelTypeEnum
    batch_time: float
    extra_info: dict

    # implement custom asdict to return a dict with the same keys as the dataclass using the model name
    def asdict(self) -> dict:
        return {
            f"{self.model_name}_raw_{self.model_type.value}": self.rewards.tolist(),
            f"{self.model_name}_{self.model_type.value}": self.rewards_normalized.tolist(),
            f"{self.model_name}_{self.model_type.value}_timings": self.timings.tolist(),
            f"{self.model_name}_{self.model_type.value}_batch_time": self.batch_time,
            f"{self.model_name}_{self.model_type.value}_extra_info": self.extra_info,
        }


class RewardResult:
    def __init__(self, reward_pipeline, agent, response_event, device):
        """Passes the responses through the reward models and calculates the total reward

        Args:
            reward_pipeline (RewardPipeline): List of all loaded/ative reward models
            task (Task): Task instance which contains reward_definition (list of reward model requirements) and a reference answer (str)
            response_event (DendriteResponseEvent): Network responses to the prompt
            device (str): Device to run the reward models on
        """

        self.reward_pipeline = reward_pipeline
        self.response_event = response_event
        self.device = device
        self.task_rewards = agent.task.reward_definition
        self.task_penalties = agent.task.penalty_definition
        self.reward_events = self.reward_responses(
            reference=agent.task.reference,
            models=self.task_rewards,
            reward_type=RewardModelTypeEnum.WEIGHTED_REWARD,
        )
        self.penalty_events = self.reward_responses(
            reference=agent.challenge,
            models=self.task_penalties,
            reward_type=RewardModelTypeEnum.PENALTY,
        )
        self.rewards = self.total_reward()

    def __state_dict__(self, full=False):
        state = {"rewards": self.rewards.tolist()}
        for event in self.reward_events + self.penalty_events:
            state.update(event.asdict())
        return state

    def reward_responses(
        self, reference: str, models: List[dict], reward_type: RewardModelTypeEnum
    ) -> List[RewardEvent]:
        """Calculates the rewards for the responses given the task and returns a RewardEvent for each reward model
        reward_events: List[RewardEvent] = [
            RewardEvent(model_name='rouge', rewards=torch.zeros(50), timings=torch.zeros(50), ...),
            RewardEvent(model_name='relevance', rewards=torch.zeros(50), timings=torch.zeros(50), ...),
        ]
        """
        reward_events = []

        for reward_info in models:
            # Select the reward model from preloaded reward model pipeline
            reward_model = self.reward_pipeline.get(reward_info["name"])
            if not reward_model:
                raise ValueError(
                    f"Reward model {reward_info['name']} not supported. Please choose from {self.reward_pipeline.reward_models.keys()}"
                )
            # Compute the rewards for the responses given the prompt
            reward_event = reward_model.apply(
                reference, self.response_event, reward_type=reward_type
            )
            reward_events.append(reward_event)

        return reward_events

    def total_reward(self) -> torch.FloatTensor:
        """Combines the rewards from all the reward models into a single reward tensor"""

        # TODO: How would using the Agent as a reward model fit into this flow?
        # Compute the rewards for the responses given the prompt
        rewards = torch.zeros_like(
            self.response_event.uids, dtype=torch.float32, device=self.device
        )

        for event in self.reward_events:
            for reward_info in filter(
                lambda x: x["name"] == event.model_name, self.task_rewards
            ):
                rewards += reward_info["weight"] * event.rewards.to(self.device)

        for event in self.penalty_events:
            for reward_info in filter(
                lambda x: x["name"] == event.model_name, self.task_penalties
            ):
                rewards *= 1 - reward_info["weight"] * event.rewards.to(self.device)

        return rewards

    def __str__(self):
        return f"{self.__class__.__name__}(rewards={self.rewards!r}, reward_events={self.reward_events!r}, penalty_events={self.penalty_events!r})"


@dataclass
class BatchRewardOutput:
    rewards: torch.FloatTensor
    timings: torch.FloatTensor
    extra_info: dict

    def __post_init__(self):
        if self.rewards.shape != self.timings.shape:
            raise ValueError(
                f"rewards.shape {self.rewards.shape} != timings.shape {self.timings.shape}"
            )

        self.rewards_normalized = (self.rewards - self.rewards.min()) / (
            self.rewards.max() - self.rewards.min() + 1e-6
        )


class BaseRewardModel(ABC):
    @property
    @abstractmethod
    def name(self) -> str:
        ...

    @abstractmethod
    def __init__(self, **kwargs):
        pass

    @abstractmethod
    def reward(self, reference: str, completions: List[str]) -> BatchRewardOutput:
        pass

    def apply(self, reference: str, response_event, reward_type) -> RewardEvent:
        t0 = time.time()
        batch_rewards_output = self.reward(reference, response_event.completions)
        batch_rewards_time = time.time() - t0

        return RewardEvent(
            model_name=self.name,
            rewards=batch_rewards_output.rewards,
            rewards_normalized=batch_rewards_output.rewards_normalized,
            model_type=reward_type,
            batch_time=batch_rewards_time,
            extra_info=batch_rewards_output.extra_info,
            timings=batch_rewards_output.timings,
        )

    def __repr__(self):
        return f"{self.__class__.__name__}(name={self.name})"
